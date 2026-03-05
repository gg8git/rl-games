"""
train_ppo.py  —  PPO for GongZhu (宫主)
═══════════════════════════════════════════════════════════════════════════════
Changelog (performance + scaling update)
─────────────────────────────────────────
[perf]  Batched opponent inference (Fix 2).
        _advance_to_ego_turn() no longer calls PPOAgent models inline.
        Instead it handles cheap non-PPO agents (RandomAgent, EpsilonAgent)
        in-place and stops when it encounters a PPOAgent opponent, setting
        self._needs_ppo_step = True.

        After each envs.step() call, the rollout loop calls
        _batch_advance_ppo_opponents() which:
          1. Collects all pending PPO requests across all NUM_ENVS environments.
          2. Groups them by agent object identity (same agent instance = same
             weights, so they can share one forward pass).
          3. Runs ONE belief pass + ONE actor pass per unique historical agent.
          4. Applies actions and resumes _advance_to_ego_turn() for each env.
          5. Repeats until no env has a pending PPO opponent turn.

        This replaces up to NUM_ENVS × 3 × 13 = 624 sequential B=1 forward
        passes per rollout step with O(pool_size) batched passes, giving a
        dramatic SPS improvement at generation ≥ 2.

[psro]  GongZhuPPOEnv now accepts an optional PolicyPool.  When provided,
        reset() samples 3 historical opponents from the pool for the episode.
        Passing policy_pool=None preserves random-opponent behaviour.

[psro]  Training logic extracted into train_ppo() for the PSRO orchestrator.

Previous changelog entries
──────────────────────────
[perf]  Batched ego belief inference (one GPU call per step across all envs).
[fix]   Sparse terminal-only reward.
[arch]  Disjoint actor / critic networks.
[arch]  Conv1d(10→32, kernel_size=1) per-card feature mixer.
[cfg]   REWARD_OPPONENT_BASELINE = "mean" | "max".
"""

from __future__ import annotations

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from belief_net import GongZhuBeliefPredictor
from base_env import GongZhuEnv, card_suit


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters (all overridable via train_ppo())
# ──────────────────────────────────────────────────────────────────────────────

EXP_NAME            = "GongZhu_PPO"
SEED                = 42
TORCH_DETERMINISTIC = True
CUDA                = True

TOTAL_TIMESTEPS   = 5_000_000
LEARNING_RATE     = 2.5e-4
ANNEAL_LR         = True
NUM_ENVS          = 16
NUM_STEPS         = 130
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_COEF         = 0.2
CLIP_VALUE_LOSS   = True
ENT_COEF          = 0.01
VF_COEF           = 0.5
MAX_GRAD_NORM     = 0.5
NORM_ADV          = True
BATCH_SIZE        = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE    = 256
UPDATE_EPOCHS     = 4

REWARD_OPPONENT_BASELINE = "mean"
TERMINAL_REWARD_SCALE    = 600.0
TERMINAL_REWARD_CLIP     = 1.5

HIDDEN_SIZE   = 512
CONV_CHANNELS = 32

BELIEF_MODEL_PATH = "./belief_net/ckpt/best.pt"

OBS_CHANNELS = 10
N_CARDS      = 52


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Gymnasium wrapper
# ──────────────────────────────────────────────────────────────────────────────

class GongZhuPPOEnv(gym.Env):
    """
    Single-agent GongZhu wrapper.

    Observation contract
    ────────────────────
    Ch 0–6: Structural features built by _get_obs() inside the env.
    Ch 7–9: Zeros — filled externally by run_batched_belief() in the rollout
            loop (one GPU call per step across all NUM_ENVS environments).

    Opponent behaviour  (Fix 2)
    ───────────────────────────
    When a PolicyPool is supplied, _advance_to_ego_turn() handles
    RandomAgent / EpsilonAgent opponents inline (cheap) and STOPS as soon as
    it hits a PPOAgent opponent, setting self._needs_ppo_step = True.

    The rollout loop detects this flag via needs_opponent_step() and calls
    _batch_advance_ppo_opponents() which batches all pending PPO requests
    across all environments before resuming via advance_ppo_opponent().

    When policy_pool is None, all opponents use random.choice (original
    standalone behaviour).
    """

    metadata = {"render_modes": []}

    def __init__(self, policy_pool=None) -> None:
        super().__init__()
        self.base_env    = GongZhuEnv()
        self.ego_player  = 0
        self.policy_pool = policy_pool
        self.opponents   = None           # set by reset() when pool is active
        self._needs_ppo_step: bool = False

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_CHANNELS, N_CARDS), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_CARDS)
        self._snapshot: dict | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.policy_pool is not None:
            self.opponents = self.policy_pool.sample_opponents(n=3)
        else:
            self.opponents = None

        self._needs_ppo_step = False
        self.base_env.reset()
        self._advance_to_ego_turn()

        # If the very first player is a PPO opponent, obs is zeros (pending).
        # The rollout loop will call _batch_advance_ppo_opponents() and then
        # refresh obs via _get_obs() / get_belief_snapshot().
        obs  = self._get_obs() if not self._needs_ppo_step else \
               np.zeros((OBS_CHANNELS, N_CARDS), dtype=np.float32)
        mask = self.get_action_mask()
        return obs, {"action_mask": mask}

    def step(self, action: int):
        if not self.base_env.done:
            legal = self.base_env.legal_actions()
            if action not in legal:
                action = random.choice(legal)

            self.base_env.step(action)
            self._needs_ppo_step = False
            self._advance_to_ego_turn()

        done = self.base_env.done
        info: dict = {}

        if done:
            scores   = self.base_env.score()
            my_score = scores[self.ego_player]
            others   = [scores[i] for i in range(4) if i != self.ego_player]
            baseline = (
                max(others) if REWARD_OPPONENT_BASELINE == "max"
                else float(np.mean(others))
            )
            reward = float(np.clip(
                (my_score - baseline) / TERMINAL_REWARD_SCALE,
                -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP,
            ))
            info["ego_score"]   = my_score
            info["others_mean"] = float(np.mean(others))
            info["is_win"]      = float(my_score >= max(others))
            info["grand_slam"]  = float(my_score >= 200)
            ego_tricks          = self.base_env.tricks_won[self.ego_player]
            info["took_pig"]    = float(49 in ego_tricks)
            info["took_sheep"]  = float(22 in ego_tricks)
        else:
            reward = 0.0

        # Return zeros obs if we're mid-opponent-chain; rollout loop will refresh.
        if done:
            obs = np.zeros((OBS_CHANNELS, N_CARDS), dtype=np.float32)
        elif self._needs_ppo_step:
            obs = np.zeros((OBS_CHANNELS, N_CARDS), dtype=np.float32)
        else:
            obs = self._get_obs()

        mask = self.get_action_mask() if not done else np.zeros(N_CARDS, dtype=np.bool_)
        info["action_mask"] = mask

        return obs, reward, done, False, info

    # ── Opponent stepping  (Fix 2) ────────────────────────────────────────────

    def _advance_to_ego_turn(self) -> None:
        """
        Advance the game until it is the ego's turn (or the game ends).

        Non-PPO opponents (RandomAgent, EpsilonAgent) are stepped inline
        because they are O(1) and have no model to load.

        PPO opponents are expensive: this method stops at the first PPO
        opponent turn and sets self._needs_ppo_step = True.  The rollout loop
        then calls _batch_advance_ppo_opponents() which groups all pending
        requests across all envs and does one batched forward pass per unique
        historical policy.
        """
        from policy_pool import PPOAgent

        while (not self.base_env.done and
               self.base_env.current_player != self.ego_player):
            cp = self.base_env.current_player

            if self.opponents is not None:
                opp_idx = (cp - self.ego_player - 1) % 4
                agent   = self.opponents[opp_idx]
                if isinstance(agent, PPOAgent):
                    # Stop here; rollout loop will batch this
                    self._needs_ppo_step = True
                    return
                else:
                    action = agent(self.base_env)
            else:
                action = random.choice(self.base_env.legal_actions())

            self.base_env.step(action)

        self._needs_ppo_step = False

    def needs_opponent_step(self) -> bool:
        """True when the env is paused waiting for a batched PPO opponent action."""
        return self._needs_ppo_step and not self.base_env.done

    def advance_ppo_opponent(self, action: int) -> None:
        """
        Apply a PPO opponent action and resume _advance_to_ego_turn().

        Called by _batch_advance_ppo_opponents() after computing the batched
        action.  May set _needs_ppo_step again if the next opponent is also
        a PPOAgent (the outer loop handles this via repeated passes).
        """
        self._needs_ppo_step = False
        self.base_env.step(action)
        self._advance_to_ego_turn()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(N_CARDS, dtype=np.bool_)
        if not self.base_env.done and not self._needs_ppo_step:
            mask[self.base_env.legal_actions()] = True
        return mask

    def get_belief_snapshot(self) -> dict:
        """Return the snapshot cached by the most recent _get_obs() call."""
        return self._snapshot  # type: ignore[return-value]

    def _get_obs(self) -> np.ndarray:
        """
        Build structural (ch 0-6) obs and cache the snapshot.
        Ch 7–9 left as zeros; filled externally by run_batched_belief().
        """
        p    = self.ego_player
        snap = self.base_env.get_snapshot(p)
        self._snapshot = snap

        obs = np.zeros((OBS_CHANNELS, N_CARDS), dtype=np.float32)
        obs[0] = snap["revealed_hands"][p]

        if len(snap["played_cards"]) > 0:
            obs[1, snap["played_cards"]] = 1.0

        if self.base_env.current_trick:
            trick_cards = [c for c, _ in self.base_env.current_trick]
            obs[2, trick_cards] = 1.0
            led_suit = card_suit(trick_cards[0])
            obs[3, led_suit * 13: led_suit * 13 + 13] = 1.0

        for rel_idx, offset in enumerate([1, 2, 3]):
            opp = (p + offset) % 4
            for suit in range(4):
                if self.base_env.voids[opp][suit]:
                    obs[4 + rel_idx, suit * 13: suit * 13 + 13] = 1.0

        return obs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PPO network
# ──────────────────────────────────────────────────────────────────────────────

def layer_init(layer: nn.Linear, std: float = np.sqrt(2),
               bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module):
    """
    Actor-Critic with disjoint networks and a Conv1d front-end per head.

    Architecture (per head)
    ───────────────────────
    (B, 10, 52) → Conv1d(10→32, k=1) → ReLU → Flatten → Linear(1664, 512)
    → Tanh → Linear(512, 256) → Tanh → head (52 or 1)
    """

    def __init__(self,
                 obs_channels: int = OBS_CHANNELS,
                 n_cards:      int = N_CARDS,
                 action_size:  int = N_CARDS,
                 hidden_size:  int = HIDDEN_SIZE,
                 conv_ch:      int = CONV_CHANNELS) -> None:
        super().__init__()
        flat_size = conv_ch * n_cards

        def make_encoder(head_out: int, head_std: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(obs_channels, conv_ch, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(flat_size, hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_size, hidden_size // 2)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_size // 2, head_out), std=head_std),
            )

        self.actor  = make_encoder(action_size, head_std=0.01)
        self.critic = make_encoder(1,           head_std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x:           torch.Tensor,
        action:      torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ):
        logits = self.actor(x)

        if torch.isnan(logits).any():
            print(
                f"[WARN] NaN in actor logits  "
                f"x: min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}  "
                f"NaN count={torch.isnan(logits).sum().item()}"
            )
            logits = torch.nan_to_num(logits, nan=0.0)

        if action_mask is not None:
            action_mask = action_mask.bool()
            empty_rows  = ~action_mask.any(dim=1)
            if empty_rows.any():
                action_mask = action_mask.clone()
                action_mask[empty_rows] = True
            logits = logits.masked_fill(~action_mask, -1e8)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_env(policy_pool=None):
    def thunk():
        env = GongZhuPPOEnv(policy_pool=policy_pool)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def extract_masks(infos: dict, envs: gym.vector.VectorEnv,
                  num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(infos, dict) and "action_mask" in infos:
        arr = np.asarray(infos["action_mask"])
    else:
        arr = np.array([
            envs.envs[i].unwrapped.get_action_mask()
            for i in range(num_envs)
        ])
    return torch.from_numpy(arr).bool().to(device)


def run_batched_belief(
    belief_model: nn.Module,
    snapshots:    list[dict],
    device:       torch.device,
) -> torch.Tensor:
    """
    Single batched belief-model forward pass over N snapshots.
    Returns (N, 3, 52) float32 — P(opp at offset +1,+2,+3 holds each card).
    """
    N        = len(snapshots)
    seq_lens = [int(s["seq_len"]) for s in snapshots]
    max_len  = max(seq_lens)

    if max_len > 0:
        pc_arr = np.zeros((N, max_len), dtype=np.int64)
        pl_arr = np.zeros((N, max_len), dtype=np.int64)
        tn_arr = np.zeros((N, max_len), dtype=np.int64)
        tp_arr = np.zeros((N, max_len), dtype=np.int64)
        for i, s in enumerate(snapshots):
            L = seq_lens[i]
            if L > 0:
                pc_arr[i, :L] = s["played_cards"]
                pl_arr[i, :L] = s["players"]
                tn_arr[i, :L] = s["trick_nums"]
                tp_arr[i, :L] = s["trick_pos"]
        pc_t = torch.from_numpy(pc_arr).to(device)
        pl_t = torch.from_numpy(pl_arr).to(device)
        tn_t = torch.from_numpy(tn_arr).to(device)
        tp_t = torch.from_numpy(tp_arr).to(device)
        sl_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
    else:
        pc_t = torch.zeros(N, 0, dtype=torch.long, device=device)
        pl_t = torch.zeros(N, 0, dtype=torch.long, device=device)
        tn_t = torch.zeros(N, 0, dtype=torch.long, device=device)
        tp_t = torch.zeros(N, 0, dtype=torch.long, device=device)
        sl_t = None

    rev_t = torch.from_numpy(
        np.stack([s["revealed_hands"] for s in snapshots])
    ).float().to(device)
    msk_t = torch.from_numpy(
        np.stack([s["mask"] for s in snapshots])
    ).float().to(device)

    with torch.no_grad():
        probs, _ = belief_model(pc_t, pl_t, tn_t, tp_t, rev_t, msk_t,
                                seq_lengths=sl_t)

    return torch.stack([probs[:, :, 1], probs[:, :, 2], probs[:, :, 3]], dim=1)


def load_belief_model(path: str, device: torch.device) -> nn.Module:
    model = GongZhuBeliefPredictor(d_model=128, n_heads=4, n_layers=3).to(device)
    if os.path.exists(path):
        chkpt = torch.load(path, map_location=device)
        if "model_state" in chkpt:
            model.load_state_dict(chkpt["model_state"])
        else:
            model.load_state_dict(chkpt)
        print(f"[belief] Loaded from {path}")
    else:
        print(f"[belief] WARNING: not found at {path} — using random weights.")
    model.eval()
    return model


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = y_true.var()
    return float(1.0 - (y_true - y_pred).var() / (var_y + 1e-8))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Batched opponent driver  (Fix 2)
# ──────────────────────────────────────────────────────────────────────────────

def _batch_advance_ppo_opponents(
    envs:       gym.vector.VectorEnv,
    num_envs:   int,
    cpu_device: torch.device,
) -> None:
    """
    Drive all pending PPO opponent turns across all envs using batched
    inference grouped by agent identity.

    Called after envs.step() and envs.reset() when a PolicyPool is active.
    Non-PPO opponents are already handled inline by _advance_to_ego_turn();
    only PPOAgent turns reach this function.

    The loop repeats until every env is either at the ego's turn or done,
    correctly handling consecutive PPO opponent turns (e.g. if both opp+1
    and opp+2 are PPOAgents in the same trick).

    Batching logic
    ──────────────
    Multiple environments may have sampled the SAME historical PPOAgent
    (same object identity = same weights).  We group by id(agent), so all
    environments waiting on the same historical policy get batched into one
    forward pass.  Environments with different historical agents each get
    their own forward pass.
    """
    from policy_pool import PPOAgent

    while True:
        # ── Collect pending requests ──────────────────────────────────────────
        pending: list[tuple] = []
        for i in range(num_envs):
            inner = envs.envs[i].unwrapped
            if inner.needs_opponent_step():
                cp      = inner.base_env.current_player
                opp_idx = (cp - inner.ego_player - 1) % 4
                agent   = inner.opponents[opp_idx]
                pending.append((i, inner, cp, agent))

        if not pending:
            break   # All envs are at ego's turn or done

        # ── Group by agent object identity ───────────────────────────────────
        # Environments that sampled the same PPOAgent instance share weights
        # and can be batched into one forward pass.
        groups: dict[int, list] = {}
        for item in pending:
            groups.setdefault(id(item[3]), []).append(item)

        # ── One batched forward pass per unique historical agent ──────────────
        for agent_id, items in groups.items():
            agent = items[0][3]
            assert isinstance(agent, PPOAgent), (
                f"Expected PPOAgent in batch loop, got {type(agent).__name__}. "
                "Non-PPO agents must be handled in _advance_to_ego_turn()."
            )

            base_envs_and_players = [
                (item[1].base_env, item[2])   # (GongZhuEnv, current_player)
                for item in items
            ]
            actions = agent.forward_batch(base_envs_and_players)

            for (env_idx, inner, cp, _), action in zip(items, actions):
                inner.advance_ppo_opponent(action)


def _refresh_obs_and_masks(
    envs:        gym.vector.VectorEnv,
    num_envs:    int,
    obs_np:      np.ndarray,
) -> np.ndarray:
    """
    After _batch_advance_ppo_opponents(), all envs are at the ego's turn (or
    done).  Rebuild obs for any env that returned dummy zeros while mid-opponent
    chain.  Done envs correctly keep their zeros.

    Also refreshes the cached belief snapshots so get_belief_snapshot() returns
    up-to-date state for the batched ego-belief pass.
    """
    for i in range(num_envs):
        inner = envs.envs[i].unwrapped
        if not inner.base_env.done:
            obs_np[i] = inner._get_obs()   # rebuilds snapshot cache too
        # done envs: zeros already set by step(); leave them
    return obs_np


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Callable training entry point
# ──────────────────────────────────────────────────────────────────────────────

def train_ppo(
    policy_pool         = None,
    belief_model_path:  str   = BELIEF_MODEL_PATH,
    total_timesteps:    int   = TOTAL_TIMESTEPS,
    save_dir:           str   = "runs",
    num_envs:           int   = NUM_ENVS,
    num_steps:          int   = NUM_STEPS,
    learning_rate:      float = LEARNING_RATE,
    anneal_lr:          bool  = ANNEAL_LR,
    minibatch_size:     int   = MINIBATCH_SIZE,
    update_epochs:      int   = UPDATE_EPOCHS,
    device_str:         str | None = None,
    seed:               int   = SEED,
    run_name:           str | None = None,
) -> str:
    """
    Run a full PPO training session and return the path to the final checkpoint.

    Returns:
        Path to agent_final.pt.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if TORCH_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True

    if device_str is None:
        device_str = "cuda" if CUDA and torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    cpu_device = torch.device("cpu")   # opponent models always live on CPU

    if run_name is None:
        run_name = f"{EXP_NAME}_{int(time.time())}"

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, "tb", run_name))

    batch_size   = num_envs * num_steps
    has_ppo_pool = (policy_pool is not None)

    # ── Setup ─────────────────────────────────────────────────────────────────
    belief_model = load_belief_model(belief_model_path, device)
    envs = gym.vector.SyncVectorEnv(
        [make_env(policy_pool) for _ in range(num_envs)]
    )

    agent     = PPOPolicy().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ── Rollout buffers ───────────────────────────────────────────────────────
    obs_shape    = envs.single_observation_space.shape   # (10, 52)
    obs_buf      = torch.zeros((num_steps, num_envs, *obs_shape), device=device)
    actions_buf  = torch.zeros((num_steps, num_envs),             device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs),             device=device)
    rewards_buf  = torch.zeros((num_steps, num_envs),             device=device)
    dones_buf    = torch.zeros((num_steps, num_envs),             device=device)
    values_buf   = torch.zeros((num_steps, num_envs),             device=device)
    masks_buf    = torch.zeros(
        (num_steps, num_envs, N_CARDS), dtype=torch.bool, device=device
    )

    # ── Initial reset ─────────────────────────────────────────────────────────
    next_obs_np, reset_infos = envs.reset(seed=seed)

    if has_ppo_pool:
        # Drive any PPO opponent turns that occur before the first ego turn
        _batch_advance_ppo_opponents(envs, num_envs, cpu_device)
        _refresh_obs_and_masks(envs, num_envs, next_obs_np)

    next_obs  = torch.from_numpy(next_obs_np).float().to(device)
    next_done = torch.zeros(num_envs, device=device)

    if has_ppo_pool:
        next_mask = torch.from_numpy(np.array([
            envs.envs[i].unwrapped.get_action_mask() for i in range(num_envs)
        ])).bool().to(device)
    else:
        next_mask = extract_masks(reset_infos, envs, num_envs, device)

    init_snaps           = [envs.envs[i].unwrapped.get_belief_snapshot()
                            for i in range(num_envs)]
    next_obs[:, 7:10, :] = run_batched_belief(belief_model, init_snaps, device)

    num_updates = total_timesteps // batch_size
    global_step = 0
    start_time  = time.time()

    episodic_returns: list[float] = []
    episodic_scores:  list[float] = []
    episodic_wins:    list[float] = []
    episodic_slams:   list[float] = []
    episodic_pigs:    list[float] = []
    episodic_sheeps:  list[float] = []

    print("=" * 80)
    print(f"  GongZhu PPO  |  device={device}  |  envs={num_envs}  |  run={run_name}")
    print(f"  batch={batch_size}  minibatch={minibatch_size}  epochs={update_epochs}")
    print(f"  pool={'PSRO' if has_ppo_pool else 'random'}  "
          f"pool_size={len(policy_pool) if has_ppo_pool else 1}  "
          f"batched_opponents={has_ppo_pool}")
    print("=" * 80)

    for update in range(1, num_updates + 1):

        lrnow = (
            (1.0 - (update - 1) / num_updates) * learning_rate
            if anneal_lr else learning_rate
        )
        optimizer.param_groups[0]["lr"] = lrnow

        # ════════════════════════════════════════════════════════════════════
        # Phase A — Rollout collection
        # ════════════════════════════════════════════════════════════════════
        for step in range(num_steps):
            global_step += num_envs

            obs_buf[step]   = next_obs
            dones_buf[step] = next_done
            masks_buf[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_mask=next_mask
                )
                values_buf[step] = value.flatten()

            actions_buf[step]  = action
            logprobs_buf[step] = logprob

            step_obs_np, reward_np, terminations, truncations, infos = \
                envs.step(action.cpu().numpy())

            rewards_buf[step] = torch.from_numpy(
                np.asarray(reward_np, dtype=np.float32)
            ).to(device)

            next_done = torch.from_numpy(
                np.logical_or(terminations, truncations).astype(np.float32)
            ).to(device)

            # ── Batch all pending PPO opponent turns  (Fix 2) ─────────────
            # Then rebuild obs for envs that returned dummy zeros mid-chain.
            if has_ppo_pool:
                _batch_advance_ppo_opponents(envs, num_envs, cpu_device)
                _refresh_obs_and_masks(envs, num_envs, step_obs_np)
                next_mask = torch.from_numpy(np.array([
                    envs.envs[i].unwrapped.get_action_mask()
                    for i in range(num_envs)
                ])).bool().to(device)
            else:
                next_mask = extract_masks(infos, envs, num_envs, device)

            next_obs = torch.from_numpy(step_obs_np).float().to(device)

            # ── Single batched ego-belief pass ────────────────────────────
            step_snaps           = [envs.envs[i].unwrapped.get_belief_snapshot()
                                    for i in range(num_envs)]
            next_obs[:, 7:10, :] = run_batched_belief(belief_model, step_snaps, device)

            # ── Episode statistics ────────────────────────────────────────
            if isinstance(infos, dict):
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if isinstance(info, dict):
                            if "episode" in info:
                                episodic_returns.append(float(info["episode"]["r"]))
                            if "ego_score" in info:
                                episodic_scores.append(float(info["ego_score"]))
                            if "is_win" in info:
                                episodic_wins.append(float(info["is_win"]))
                                episodic_slams.append(float(info["grand_slam"]))
                                episodic_pigs.append(float(info["took_pig"]))
                                episodic_sheeps.append(float(info["took_sheep"]))
                elif terminations.any() or truncations.any():
                    term_mask = np.logical_or(terminations, truncations)
                    if "episode" in infos and isinstance(infos["episode"], dict):
                        for i, is_done in enumerate(term_mask):
                            if is_done:
                                episodic_returns.append(float(infos["episode"]["r"][i]))
                    if "ego_score" in infos:
                        for i, is_done in enumerate(term_mask):
                            if is_done:
                                episodic_scores.append(float(infos["ego_score"][i]))
                    if "is_win" in infos:
                        for i, is_done in enumerate(term_mask):
                            if is_done:
                                episodic_wins.append(float(infos["is_win"][i]))
                                episodic_slams.append(float(infos["grand_slam"][i]))
                                episodic_pigs.append(float(infos["took_pig"][i]))
                                episodic_sheeps.append(float(infos["took_sheep"][i]))

        # ════════════════════════════════════════════════════════════════════
        # Phase B — GAE advantage estimation
        # ════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            next_value   = agent.get_value(next_obs).reshape(1, -1)
            advantages   = torch.zeros_like(rewards_buf)
            last_gae_lam = 0.0

            for t in reversed(range(num_steps)):
                nextnonterminal = (
                    1.0 - next_done if t == num_steps - 1
                    else 1.0 - dones_buf[t + 1]
                )
                nextvalues = (
                    next_value if t == num_steps - 1
                    else values_buf[t + 1]
                )
                delta         = (rewards_buf[t]
                                 + GAMMA * nextvalues * nextnonterminal
                                 - values_buf[t])
                advantages[t] = last_gae_lam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * last_gae_lam
                )

            returns = advantages + values_buf

        # ════════════════════════════════════════════════════════════════════
        # Phase C — Policy optimisation
        # ════════════════════════════════════════════════════════════════════
        b_obs        = obs_buf.reshape((-1, *obs_shape))
        b_logprobs   = logprobs_buf.reshape(-1)
        b_actions    = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values_buf.reshape(-1)
        b_masks      = masks_buf.reshape((-1, N_CARDS))

        b_inds    = np.arange(batch_size)
        clipfracs: list[float] = []
        pg_total = v_total = ent_total = kl_total = 0.0
        n_mb = 0

        for _epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start: start + minibatch_size]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds],
                    action=b_actions.long()[mb_inds],
                    action_mask=b_masks[mb_inds],
                )

                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio     = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if CLIP_VALUE_LOSS:
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds], -CLIP_COEF, CLIP_COEF
                    )
                    v_loss = 0.5 * torch.max(
                        (new_value - b_returns[mb_inds]) ** 2,
                        (v_clipped  - b_returns[mb_inds]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                pg_total  += pg_loss.item()
                v_total   += v_loss.item()
                ent_total += entropy_loss.item()
                kl_total  += approx_kl.item()
                n_mb      += 1

        avg_pg  = pg_total  / n_mb
        avg_v   = v_total   / n_mb
        avg_ent = ent_total / n_mb
        avg_kl  = kl_total  / n_mb
        exp_var = explained_variance(b_values, b_returns)

        # ════════════════════════════════════════════════════════════════════
        # Phase D — Logging & checkpointing
        # ════════════════════════════════════════════════════════════════════
        if update % 5 == 0:
            sps         = int(global_step / (time.time() - start_time))
            mean_return = float(np.mean(episodic_returns)) if episodic_returns else 0.0
            mean_score  = float(np.mean(episodic_scores))  if episodic_scores  else 0.0
            win_rate    = float(np.mean(episodic_wins))    if episodic_wins    else 0.0
            slam_rate   = float(np.mean(episodic_slams))   if episodic_slams   else 0.0
            pig_rate    = float(np.mean(episodic_pigs))    if episodic_pigs    else 0.0
            sheep_rate  = float(np.mean(episodic_sheeps))  if episodic_sheeps  else 0.0

            episodic_returns.clear(); episodic_scores.clear()
            episodic_wins.clear();    episodic_slams.clear()
            episodic_pigs.clear();    episodic_sheeps.clear()

            print(
                f"Update {update:04d}/{num_updates}  "
                f"Ret={mean_return:6.3f}  Score={mean_score:6.1f}  "
                f"Win={win_rate:5.1%}  Pig={pig_rate:5.1%}  Sheep={sheep_rate:5.1%}  "
                f"PG={avg_pg:.4f}  V={avg_v:.4f}  "
                f"Ent={avg_ent:.4f}  KL={avg_kl:.4f}  "
                f"EV={exp_var:.3f}  SPS={sps}"
            )

            writer.add_scalar("charts/learning_rate",       lrnow,              global_step)
            writer.add_scalar("charts/SPS",                 sps,                global_step)
            writer.add_scalar("losses/policy_loss",         avg_pg,             global_step)
            writer.add_scalar("losses/value_loss",          avg_v,              global_step)
            writer.add_scalar("losses/entropy",             avg_ent,            global_step)
            writer.add_scalar("losses/approx_kl",           avg_kl,             global_step)
            writer.add_scalar("losses/clipfrac",            np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance",  exp_var,            global_step)
            writer.add_scalar("rollout/mean_return",        mean_return,        global_step)
            writer.add_scalar("rollout/mean_ego_score",     mean_score,         global_step)
            writer.add_scalar("rollout/win_rate",           win_rate,           global_step)
            writer.add_scalar("rollout/slam_rate",          slam_rate,          global_step)
            writer.add_scalar("rollout/pig_capture_rate",   pig_rate,           global_step)
            writer.add_scalar("rollout/sheep_capture_rate", sheep_rate,         global_step)

        if update % 50 == 0:
            ckpt_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "update":          update,
                    "global_step":     global_step,
                    "agent_state":     agent.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, f"ckpt_update_{update:06d}.pt"),
            )

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "agent_final.pt")
    torch.save(
        {
            "update":      num_updates,
            "global_step": global_step,
            "agent_state": agent.state_dict(),
        },
        final_path,
    )
    envs.close()
    writer.close()
    print(f"Training complete. Agent saved to {final_path}")
    return final_path


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_ppo(
        policy_pool       = None,
        belief_model_path = BELIEF_MODEL_PATH,
        total_timesteps   = TOTAL_TIMESTEPS,
        save_dir          = f"runs/{EXP_NAME}_{int(time.time())}",
    )
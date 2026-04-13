"""
policy_pool.py
──────────────
Manages the growing population of historical agents used in PSRO.

Classes
───────
BaseAgent       — Abstract base; all agents implement __call__(env) -> int
RandomAgent     — Uniform random over legal actions
EpsilonAgent    — ε-greedy wrapper around any base agent
_ModelCache     — LRU cache for PPO + Belief model pairs (RAM-bounded)
PPOAgent        — Path-only wrapper; models loaded lazily via _ModelCache
PolicyPool      — Container + sampling logic for the PSRO population

RAM management (Fix 1)
──────────────────────
The naive approach of calling PPOPolicy() + load_belief_model() in
PPOAgent.__init__ loads every historical network into RAM permanently.
By Generation 15 you hold 15 Transformers + 15 PPO actors simultaneously —
easily 10–20 GB, triggering OOM when 16 parallel training envs are active.

Solution: PPOAgent stores only (ppo_path, belief_path, device).  A module-
level _ModelCache backed by collections.OrderedDict evicts the least-recently-
used model pair whenever the cache exceeds its capacity.  Default capacity=4
covers the 3 opponent slots per episode plus one buffer for newly-sampled
agents, keeping peak RAM flat regardless of pool size.

Batched inference (Fix 2)
─────────────────────────
PPOAgent.forward_batch(base_envs_and_players) accepts a list of (GongZhuEnv,
player_idx) pairs and runs ONE belief forward pass + ONE actor forward pass for
the entire batch.  The training loop calls this instead of __call__ so that all
PPO opponent moves across all NUM_ENVS environments are batched per unique
historical policy, replacing hundreds of B=1 sequential CPU calls with one
call per unique agent per rollout step.
"""

from __future__ import annotations

import gc
import random
from collections import OrderedDict
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from base_env import GongZhuEnv, card_suit


# ──────────────────────────────────────────────────────────────────────────────
# Base classes
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """All agents share this interface."""

    def __call__(self, env: GongZhuEnv) -> int:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Selects uniformly at random from all legal actions."""

    def __call__(self, env: GongZhuEnv) -> int:
        return random.choice(env.legal_actions())


class EpsilonAgent(BaseAgent):
    """
    ε-greedy wrapper around any base agent.

    With probability ε    → plays a random legal card.
    With probability 1-ε  → delegates to base(env).
    """

    def __init__(self, base: Callable, epsilon: float) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be in [0, 1]"
        self.base    = base
        self.epsilon = epsilon

    def __call__(self, env: GongZhuEnv) -> int:
        if random.random() < self.epsilon:
            return random.choice(env.legal_actions())
        return self.base(env)


# ──────────────────────────────────────────────────────────────────────────────
# LRU model cache  (Fix 1)
# ──────────────────────────────────────────────────────────────────────────────

class _ModelCache:
    """
    LRU cache for (PPOPolicy, GongZhuBeliefPredictor) model pairs.

    Keeps at most `capacity` pairs in RAM.  When a new pair is requested and
    the cache is full, the least-recently-used pair is deleted and Python's
    garbage collector is invoked to release memory immediately.

    Keys are (ppo_path, belief_path) tuples so each unique historical
    checkpoint pair gets exactly one copy in memory at a time.

    Thread safety: not guaranteed — GongZhuPPOEnv runs inside SyncVectorEnv
    which is single-threaded, so this is fine.
    """

    def __init__(self, capacity: int = 5) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[tuple, tuple[nn.Module, nn.Module]] = OrderedDict()

    def get(
        self,
        ppo_path:    str,
        belief_path: str,
        device:      torch.device,
    ) -> tuple[nn.Module, nn.Module]:
        """
        Return the (ppo_model, belief_model) pair for this checkpoint combo,
        loading from disk if necessary.  Touches the entry so it becomes MRU.
        """
        key = (ppo_path, belief_path)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        # ── Load from disk ────────────────────────────────────────────────────
        from train_ppo import PPOPolicy, load_belief_model

        ppo_model = PPOPolicy().to(device)
        ckpt      = torch.load(ppo_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("agent_state", ckpt)
        else:
            state_dict = ckpt
        ppo_model.load_state_dict(state_dict)
        ppo_model.eval()

        belief_model = load_belief_model(belief_path, device)

        pair = (ppo_model, belief_model)
        self._cache[key] = pair
        self._cache.move_to_end(key)

        # ── Evict LRU if over capacity ────────────────────────────────────────
        while len(self._cache) > self.capacity:
            evict_key, evict_pair = self._cache.popitem(last=False)
            del evict_pair          # drop references
            gc.collect()            # free ASAP; avoids silent OOM growth
            print(
                f"[ModelCache] Evicted {evict_key[0]!r}. "
                f"Cache size: {len(self._cache)}/{self.capacity}"
            )

        return pair

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Flush all cached models (e.g. between PSRO generations)."""
        self._cache.clear()
        gc.collect()


# Module-level singleton — shared across all PPOAgent instances.
# Capacity=4: covers 3 opponent slots per episode + 1 spare for newly-sampled
# agents that haven't been evicted yet.
_MODEL_CACHE = _ModelCache(capacity=8)


# ──────────────────────────────────────────────────────────────────────────────
# PPO agent (Fix 1 + Fix 2)
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent(BaseAgent):
    """
    Lazy-loading wrapper around a saved PPO + Belief checkpoint pair.

    __init__ stores only file paths — no model is loaded until the first
    inference call.  Models are fetched from the module-level _MODEL_CACHE
    which evicts the LRU pair when capacity is exceeded, keeping peak RAM
    constant regardless of pool size.

    forward_batch() is the primary inference path used by the training loop.
    It accepts a list of (GongZhuEnv, player_idx) pairs and runs one batched
    belief pass + one batched actor pass, replacing N sequential B=1 calls
    with a single B=N call.

    __call__() is preserved for backward compatibility (used by eval_meta_game
    which calls agents sequentially from Python).

    Args:
        ppo_path:    Path to the PPO checkpoint (.pt).
        belief_path: Path to the belief-net checkpoint (.pt) frozen at the
                     time this policy was trained.
        device:      Torch device string, default "cpu".
    """

    CHANNELS = 10
    CARDS    = 52

    def __init__(self, ppo_path: str, belief_path: str, device: str = "cpu") -> None:
        # Lazy import — avoids circular import at module load time
        from train_ppo import run_batched_belief
        self._run_batched_belief = run_batched_belief

        self.ppo_path    = ppo_path
        self.belief_path = belief_path
        self.device      = torch.device(device)
        # No model loading here — deferred to first use via _MODEL_CACHE

    @property
    def _models(self) -> tuple[nn.Module, nn.Module]:
        """Fetch (ppo_model, belief_model) from the LRU cache, loading if needed."""
        return _MODEL_CACHE.get(self.ppo_path, self.belief_path, self.device)

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs_batch(
        self,
        base_envs_and_players: list[tuple[GongZhuEnv, int]],
    ) -> tuple[np.ndarray, list[dict], list[list[int]]]:
        """
        Build the (B, 7, 52) structural observation array (channels 0-6) for a
        batch of (GongZhuEnv, player_idx) pairs, returning it alongside the
        snapshots (needed for belief inference) and legal action lists.

        Channel layout mirrors GongZhuPPOEnv._get_obs():
          Ch 0  ego hand           Ch 4  opp+1 voids
          Ch 1  all played cards   Ch 5  opp+2 voids
          Ch 2  current trick      Ch 6  opp+3 voids
          Ch 3  led-suit mask      Ch 7-9 zeros (filled by belief pass)
        """
        B     = len(base_envs_and_players)
        obs   = np.zeros((B, self.CHANNELS, self.CARDS), dtype=np.float32)
        snaps: list[dict]       = []
        legal: list[list[int]]  = []

        for i, (base_env, player_idx) in enumerate(base_envs_and_players):
            snap = base_env.get_snapshot(player_idx)
            snaps.append(snap)
            legal.append(base_env.legal_actions(player_idx))

            obs[i, 0] = snap["revealed_hands"][player_idx]

            if len(snap["played_cards"]) > 0:
                obs[i, 1, snap["played_cards"]] = 1.0

            if base_env.current_trick:
                trick_cards = [c for c, _ in base_env.current_trick]
                obs[i, 2, trick_cards] = 1.0
                led_suit = card_suit(trick_cards[0])
                obs[i, 3, led_suit * 13: led_suit * 13 + 13] = 1.0

            for rel_idx, offset in enumerate([1, 2, 3]):
                opp = (player_idx + offset) % 4
                for suit in range(4):
                    if base_env.voids[opp][suit]:
                        obs[i, 4 + rel_idx, suit * 13: suit * 13 + 13] = 1.0

        return obs, snaps, legal

    # ── Inference ─────────────────────────────────────────────────────────────

    def forward_batch(
        self,
        base_envs_and_players: list[tuple[GongZhuEnv, int]],
    ) -> list[int]:
        """
        Run a single batched forward pass for B (env, player_idx) pairs.

        This is the primary inference path called by the training loop after
        collecting all pending PPO opponent requests across all envs.  Running
        one belief pass (B=N) + one actor pass (B=N) replaces N sequential
        B=1 calls.

        Args:
            base_envs_and_players: List of (GongZhuEnv, player_idx) pairs.

        Returns:
            List of B integer card actions (one per pair).
        """
        ppo_model, belief_model = self._models

        obs_np, snaps, legal_lists = self._build_obs_batch(base_envs_and_players)

        # Single batched belief pass for all B environments
        belief_channels     = self._run_batched_belief(belief_model, snaps, self.device)
        obs_np[:, 7:10, :]  = belief_channels.cpu().numpy()

        obs_t = torch.from_numpy(obs_np).to(self.device)

        # Build action masks
        B      = len(base_envs_and_players)
        masks  = np.zeros((B, self.CARDS), dtype=np.bool_)
        for i, la in enumerate(legal_lists):
            masks[i, la] = True
        mask_t = torch.from_numpy(masks).to(self.device)

        with torch.no_grad():
            logits  = ppo_model.actor(obs_t)
            logits  = logits.masked_fill(~mask_t, -1e8)
            actions = logits.argmax(dim=-1)

        return actions.cpu().tolist()

    def __call__(self, env: GongZhuEnv) -> int:
        """
        Single-env greedy action (backward-compatible interface).

        Used by eval_meta_game.simulate_matchup() which calls agents
        sequentially.  The training loop uses forward_batch() instead.
        """
        return self.forward_batch([(env, env.current_player)])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Policy pool
# ──────────────────────────────────────────────────────────────────────────────

class PolicyPool:
    """
    Maintains the growing PSRO population and provides sampling utilities.

    The pool is always seeded with a RandomAgent at index 0 so that early
    training never faces an empty sample space.

    Usage
    ─────
        pool = PolicyPool()                                  # gen-0 bootstrap
        pool.add_policy("gen1/ppo_final.pt", "belief.pt")   # after gen-1 PPO
        opponents = pool.sample_opponents(n=3)               # for env reset
    """

    def __init__(self) -> None:
        self.population:   List[BaseAgent]    = [RandomAgent()]
        self.distribution: np.ndarray | None  = None   # AlphaRank weights

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_policy(
        self,
        ppo_path:    str,
        belief_path: str,
        device:      str = "cpu",
    ) -> None:
        """
        Register a new PPO checkpoint in the pool.

        The PPOAgent is constructed with paths only (no model loading).
        The LRU cache loads weights on first use and evicts stale models
        automatically.
        """
        agent = PPOAgent(ppo_path, belief_path, device=device)
        self.population.append(agent)
        print(
            f"[PolicyPool] Added PPOAgent from '{ppo_path}'. "
            f"Pool size: {len(self.population)}"
        )

    def add_random(self) -> None:
        """Append an extra RandomAgent (rarely needed beyond gen-0)."""
        self.population.append(RandomAgent())

    def set_distribution(self, distribution: np.ndarray) -> None:
        """
        Store an AlphaRank meta-Nash distribution over the pool.

        The distribution must have the same length as self.population.
        It is normalised to sum to 1 so small floating-point errors in the
        solver output don't cause np.random.choice to raise.
        """
        distribution = np.asarray(distribution, dtype=np.float64)
        if len(distribution) != len(self.population):
            raise ValueError(
                f"Distribution length {len(distribution)} does not match "
                f"pool size {len(self.population)}."
            )
        total = distribution.sum()
        if total <= 0:
            raise ValueError("Distribution must contain at least one positive entry.")
        self.distribution = distribution / total

    def clear_distribution(self) -> None:
        """Reset to uniform sampling (e.g. after pool grows and old dist is stale)."""
        self.distribution = None

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_opponents(self, n: int = 3) -> List[BaseAgent]:
        """
        Sample n agents from the pool (with replacement).

        When self.distribution is None, samples uniformly — correct for the
        first generation before any AlphaRank evaluation.

        When self.distribution is set, samples proportionally to the meta-Nash
        weights so training targets the empirical best response against the
        current meta.
        """
        if self.distribution is None:
            return [random.choice(self.population) for _ in range(n)]

        indices = np.random.choice(
            len(self.population), size=n, p=self.distribution, replace=True
        )
        return [self.population[i] for i in indices]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.population)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for agent in self.population:
            name = type(agent).__name__
            counts[name] = counts.get(name, 0) + 1
        breakdown = ", ".join(f"{k}×{v}" for k, v in counts.items())
        dist_info = "AlphaRank" if self.distribution is not None else "uniform"
        return (
            f"<PolicyPool size={len(self.population)} "
            f"[{breakdown}] sampling={dist_info}>"
        )
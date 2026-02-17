"""
Best-of-Both-Worlds: PPO + SFT Implementation for Gridlock

Combines the best features from both versions:
- Defensive NaN checking (regular version)
- Simpler deadlock handling (v2 version)  
- Fixed device references
- Better safety checks
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym

# Import your env
from gridlock.cleanrl_gym_env import GridlockEnv

# --- Hyperparameters ---
EXP_NAME = "Gridlock_PPO_SFT"
SEED = 1
TORCH_DETERMINISTIC = True
CUDA = True

# Training Config
TOTAL_TIMESTEPS = 5000000
LEARNING_RATE = 2.5e-4
NUM_ENVS = 8
NUM_STEPS = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)
MINIBATCH_SIZE = 64
UPDATE_EPOCHS = 4

# SFT Config
DO_SFT = True
SFT_DEMOS = 5000
SFT_THRESHOLD = 20
SFT_EPOCHS = 10
SFT_LEARNING_RATE = 3e-4  # Lower than PPO LR to prevent instability

# Evaluation Config
EVAL_INTERVAL = 10
PATIENCE = 50

# Network Config
HIDDEN_SIZE = 128

def make_env():
    def thunk():
        env = GridlockEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, hidden_size=128):
        super().__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        action_size = envs.single_action_space.n
        
        # CRITIC
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        # ACTOR
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_mask=None):
        logits = self.actor(x)
        
        # NaN check before masking (defensive programming)
        if torch.isnan(logits).any():
            print(f"WARNING: NaN detected in actor logits before masking!")
            print(f"Input x stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
            print(f"NaN count: {torch.isnan(logits).sum().item()}/{logits.numel()}")
            # Replace NaN with 0 as emergency fallback
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        # --- Invalid Action Masking ---
        if action_mask is not None:
            # Ensure mask is boolean
            action_mask = action_mask.bool()
            
            # Check for deadlocked environments (no valid actions)
            # This is a normal game condition - some card orderings lead to early termination
            is_row_empty = ~action_mask.any(dim=1)
            
            if is_row_empty.any():
                # For deadlocked envs, unmask everything
                # Agent will sample any action, environment will terminate with final score
                # This is simpler than conditional masking and works correctly
                action_mask[is_row_empty] = True
            
            # Apply mask: valid actions keep their logits, invalid get -1e8
            logits = torch.where(
                action_mask, 
                logits, 
                torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            )
        
        # NaN check after masking (catch any issues from masking operation)
        if torch.isnan(logits).any():
            print(f"WARNING: NaN detected in actor logits after masking!")
            logits = torch.where(torch.isnan(logits), torch.tensor(-1e8, device=logits.device, dtype=logits.dtype), logits)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# --- Helper function to extract masks from vectorized envs ---
def get_action_masks_from_infos(infos):
    """
    Extract action masks from the info dicts returned by vectorized environments.
    
    Note: After reset() or step(), the info dict contains 'action_mask' for each env.
    For vectorized envs, we need to handle the structure properly.
    """
    # gym.vector.SyncVectorEnv returns infos as a dict with env-specific keys
    # We need to extract action_mask from each environment's info
    
    masks = []
    
    # Handle different info structures
    if isinstance(infos, dict):
        # Check if it's the new structure with numbered keys
        if 'action_mask' in infos:
            # Single environment or unified structure
            return np.array([infos['action_mask']])
        else:
            # Vectorized structure - infos might have per-env data
            # Try to get masks from the dict values
            for key in sorted([k for k in infos.keys() if isinstance(k, int) or k.isdigit()]):
                if 'action_mask' in infos[key]:
                    masks.append(infos[key]['action_mask'])
    elif isinstance(infos, (list, tuple)):
        # List of info dicts (older gymnasium versions)
        for info in infos:
            if isinstance(info, dict) and 'action_mask' in info:
                masks.append(info['action_mask'])
    
    # If we couldn't extract masks from infos, fall back to calling get_action_mask directly
    if len(masks) == 0:
        # This is the fallback - directly access the environment
        return get_action_masks_from_envs_direct(infos)
    
    return np.array(masks)

def get_action_masks_from_envs_direct(envs):
    """
    Fallback: Extract action masks by directly calling get_action_mask on unwrapped envs.
    """
    masks = []
    for i in range(envs.num_envs):
        env = envs.envs[i]
        # Unwrap to get to the actual GridlockEnv
        while hasattr(env, 'env'):
            env = env.env
        masks.append(env.get_action_mask())
    return np.array(masks)

# --- SFT Helper Functions ---
def collect_expert_data(min_score=20, num_games=1000, seed=1):
    """Collect expert demonstrations with proper seeding"""
    print(f"Generating {num_games} expert demos (Score >= {min_score})...")
    env = GridlockEnv()
    obs_list, act_list = [], []
    
    collected = 0
    attempts = 0
    rng = np.random.RandomState(seed)
    max_attempts = num_games * 10
    
    while collected < num_games and attempts < max_attempts:
        obs, info = env.reset(seed=seed + attempts)
        attempts += 1
        done = False
        temp_obs, temp_act = [], []
        
        while not done:
            mask = info["action_mask"]
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0: 
                break
            
            action = rng.choice(valid_indices)
            
            temp_obs.append(obs)
            temp_act.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done and reward >= min_score:
                obs_list.extend(temp_obs)
                act_list.extend(temp_act)
                collected += 1
                if collected % 500 == 0: 
                    print(f"  Collected {collected}/{num_games} (attempts: {attempts})")
    
    if collected < num_games:
        print(f"Warning: Only collected {collected}/{num_games} after {attempts} attempts")
    else:
        print(f"Collection complete after {attempts} attempts")
    
    return np.array(obs_list), np.array(act_list)

def evaluate_policy(agent, env, n_eval=50, device=None, argmax=False):
    """
    Evaluate policy over n_eval episodes
    
    Args:
        agent: The agent to evaluate
        env: Single environment (not vectorized)
        n_eval: Number of evaluation episodes
        device: Device to run on
        argmax: If True, use argmax action selection; if False, sample from policy
    
    Returns:
        mean_score, success_rate, mean_length
    """
    if device is None:
        device = next(agent.parameters()).device
    
    # Unwrap the environment to access get_action_mask
    eval_env = env
    while hasattr(eval_env, 'env'):
        eval_env = eval_env.env
    
    scores = []
    episode_lengths = []
    
    for _ in range(n_eval):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask = torch.tensor(eval_env.get_action_mask()).unsqueeze(0).to(device)
                
                if argmax:
                    logits = agent.actor(obs_tensor)
                    logits = torch.where(mask, logits, torch.tensor(float('-inf')).to(device))
                    action = logits.argmax(dim=-1)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_tensor, action_mask=mask)
                
            obs, reward, terminated, truncated, info = eval_env.step(action.item())
            done = terminated or truncated
            steps += 1
            if done:
                episode_reward = reward
        
        if episode_reward > 0:
            scores.append(episode_reward)
        episode_lengths.append(steps)
    
    mean_score = np.mean(scores) if scores else 0.0
    mean_length = np.mean(episode_lengths)
    success_rate = len(scores) / n_eval
    
    return mean_score, success_rate, mean_length

if __name__ == "__main__":
    # Seed everything for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = TORCH_DETERMINISTIC
    
    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    run_name = f"{EXP_NAME}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Write hyperparameters to TensorBoard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in {
            'exp_name': EXP_NAME,
            'seed': SEED,
            'total_timesteps': TOTAL_TIMESTEPS,
            'learning_rate': LEARNING_RATE,
            'num_envs': NUM_ENVS,
            'hidden_size': HIDDEN_SIZE,
        }.items()])),
    )

    # Env Setup
    envs = gym.vector.SyncVectorEnv([make_env() for i in range(NUM_ENVS)])
    agent = Agent(envs, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ==========================================
    # PHASE 1: Supervised Fine Tuning (SFT)
    # ==========================================
    if DO_SFT:
        print("\n" + "="*80)
        print("PHASE 1: SUPERVISED FINE-TUNING")
        print("="*80)
        
        # 1. Collect Expert Data
        expert_obs, expert_acts = collect_expert_data(
            min_score=SFT_THRESHOLD, 
            num_games=SFT_DEMOS, 
            seed=SEED
        )
        
        if len(expert_obs) == 0:
            print("ERROR: No expert demonstrations collected! Exiting.")
            exit(1)
        
        # 2. Create Dataset
        dataset = TensorDataset(
            torch.tensor(expert_obs, dtype=torch.float32).to(device), 
            torch.tensor(expert_acts, dtype=torch.long).to(device)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Create separate optimizer for SFT with lower learning rate
        sft_optimizer = optim.Adam(agent.parameters(), lr=SFT_LEARNING_RATE, eps=1e-5)
        
        # 3. Train Actor
        print(f"Training Actor on {len(expert_obs)} states for {SFT_EPOCHS} epochs...")
        print(f"SFT Learning Rate: {SFT_LEARNING_RATE}")
        for epoch in range(SFT_EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            
            for b_obs, b_act in loader:
                sft_optimizer.zero_grad()
                logits = agent.actor(b_obs)
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    print(f"ERROR: NaN in logits during SFT epoch {epoch+1}")
                    print(f"Batch obs stats: min={b_obs.min():.4f}, max={b_obs.max():.4f}")
                    print("Skipping this batch...")
                    continue
                
                loss = nn.functional.cross_entropy(logits, b_act)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"ERROR: NaN loss during SFT epoch {epoch+1}")
                    print("Skipping this batch...")
                    continue
                
                loss.backward()
                
                # Gradient clipping during SFT
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                
                sft_optimizer.step()
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == b_act).sum().item()
                total += b_act.size(0)
            
            if total == 0:
                print(f"ERROR: No valid batches in epoch {epoch+1}!")
                break
            
            epoch_loss = total_loss / len(loader)
            epoch_acc = correct / total
            print(f"  Epoch {epoch+1}/{SFT_EPOCHS}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.2%}")
            
            writer.add_scalar("sft/loss", epoch_loss, epoch)
            writer.add_scalar("sft/accuracy", epoch_acc, epoch)
        
        print("SFT Training Complete!\n")
        
        # Check network health after SFT
        print("Checking network health...")
        with torch.no_grad():
            test_obs = torch.randn(10, 10, device=device) * 0.1 + 0.5  # Random obs in [0,1] range
            test_logits = agent.actor(test_obs)
            test_values = agent.critic(test_obs)
            
            if torch.isnan(test_logits).any() or torch.isinf(test_logits).any():
                print("ERROR: Network producing NaN/Inf after SFT!")
                print(f"Logits: {test_logits[0]}")
                print("ABORTING - Network is broken")
                exit(1)
            
            print(f"✓ Actor output range: [{test_logits.min():.4f}, {test_logits.max():.4f}]")
            print(f"✓ Critic output range: [{test_values.min():.4f}, {test_values.max():.4f}]")
        
        # Validate SFT Performance
        print("Validating SFT performance...")
        # Get unwrapped environment for evaluation
        eval_env = envs.envs[0]
        sft_score, sft_success, sft_length = evaluate_policy(
            agent, eval_env, n_eval=100, device=device, argmax=False
        )
        print(f"SFT Evaluation (sampling):")
        print(f"  Mean Score: {sft_score:.2f}")
        print(f"  Success Rate: {sft_success:.1%}")
        print(f"  Mean Episode Length: {sft_length:.1f}")
        
        writer.add_scalar("sft/eval_score", sft_score, 0)
        writer.add_scalar("sft/eval_success_rate", sft_success, 0)
        writer.add_scalar("sft/eval_episode_length", sft_length, 0)
        
        # Reset optimizer state for PPO
        print("Resetting optimizer for PPO phase...\n")
        optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ==========================================
    # PHASE 2: PPO Training
    # ==========================================
    print("="*80)
    print("PHASE 2: PPO TRAINING")
    print("="*80 + "\n")
    
    # Storage setup
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    masks = torch.zeros((NUM_STEPS, NUM_ENVS, envs.single_action_space.n), dtype=torch.bool).to(device)

    # Start game
    global_step = 0
    start_time = time.time()
    next_obs, reset_infos = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    # Extract masks from reset info
    # For SyncVectorEnv, reset returns obs and a dict of infos
    # We need to extract action_mask from each environment
    initial_masks = []
    for i in range(NUM_ENVS):
        # Get the unwrapped environment to call get_action_mask
        env = envs.envs[i]
        while hasattr(env, 'env'):
            env = env.env
        initial_masks.append(env.get_action_mask())
    next_mask = torch.tensor(np.array(initial_masks), dtype=torch.bool).to(device)

    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE
    
    # Early stopping tracking
    best_eval_score = float('-inf')
    patience_counter = 0

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        # ============================================
        # ROLLOUT PHASE: Collect experience
        # ============================================
        for step in range(0, NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_mask=next_mask
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute game step
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)
            
            # Extract masks from each environment
            step_masks = []
            for i in range(NUM_ENVS):
                env = envs.envs[i]
                while hasattr(env, 'env'):
                    env = env.env
                step_masks.append(env.get_action_mask())
            next_mask = torch.tensor(np.array(step_masks), dtype=torch.bool).to(device)

            # Episode logging with RecordEpisodeStatistics wrapper
            # Note: gym.vector.SyncVectorEnv returns infos as dict with 'final_info' key
            if isinstance(infos, dict) and "final_info" in infos:
                for idx, info in enumerate(infos["final_info"]):
                    if info is not None and "episode" in info:
                        print(f"global_step={global_step}, env={idx}, "
                              f"episodic_return={info['episode']['r']:.1f}, "
                              f"episodic_length={info['episode']['l']}")
                        writer.add_scalar("rollout/episodic_return", 
                                        info['episode']['r'], global_step)
                        writer.add_scalar("rollout/episodic_length", 
                                        info['episode']['l'], global_step)

        # ============================================
        # ADVANTAGE COMPUTATION: Bootstrap value if not done
        # ============================================
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # ============================================
        # OPTIMIZATION PHASE: Update policy
        # ============================================
        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, envs.single_action_space.n))

        # Optimization loop
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    action=b_actions.long()[mb_inds],
                    action_mask=b_masks[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (PPO clipped objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # ============================================
        # LOGGING
        # ============================================
        if update % 10 == 0:
            mean_reward = rewards.sum().item() / NUM_ENVS
            sps = int(global_step / (time.time() - start_time))
            
            print(f"Update {update:4d}/{num_updates} | "
                  f"Mean Reward: {mean_reward:7.2f} | "
                  f"Value Loss: {v_loss.item():.4f} | "
                  f"Policy Loss: {pg_loss.item():.4f} | "
                  f"SPS: {sps}")
            
            writer.add_scalar("charts/learning_rate", lrnow, global_step)
            writer.add_scalar("charts/update", update, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("rollout/mean_reward", mean_reward, global_step)
        
        # ============================================
        # EVALUATION & EARLY STOPPING
        # ============================================
        if update % EVAL_INTERVAL == 0:
            eval_env = envs.envs[0]
            eval_score, eval_success, eval_length = evaluate_policy(
                agent, eval_env, n_eval=50, device=device, argmax=False
            )
            
            writer.add_scalar("eval/mean_score", eval_score, global_step)
            writer.add_scalar("eval/success_rate", eval_success, global_step)
            writer.add_scalar("eval/episode_length", eval_length, global_step)
            
            print(f"  >>> Evaluation: Score={eval_score:.2f}, "
                  f"Success={eval_success:.1%}, "
                  f"Length={eval_length:.1f}")
            print(f"  >>> Best Score: {best_eval_score:.2f}")
            
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                patience_counter = 0
                save_path = f"runs/{run_name}/best_agent.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(agent.state_dict(), save_path)
                print(f"  >>> NEW BEST! Saved to {save_path}")
            else:
                patience_counter += 1
                print(f"  >>> No improvement ({patience_counter}/{PATIENCE})")
                
                if patience_counter >= PATIENCE:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING at update {update}")
                    print(f"No improvement for {PATIENCE} evaluations")
                    print(f"Best score: {best_eval_score:.2f}")
                    print(f"{'='*80}\n")
                    break

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model if we have one
    best_model_path = f"runs/{run_name}/best_agent.pt"
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        agent.load_state_dict(torch.load(best_model_path))
    
    eval_env = envs.envs[0]
    
    # Evaluate with sampling
    final_score_sampling, final_success_sampling, final_length_sampling = evaluate_policy(
        agent, eval_env, n_eval=200, device=device, argmax=False
    )
    print(f"\nFinal Evaluation (Sampling from policy):")
    print(f"  Mean Score: {final_score_sampling:.2f}")
    print(f"  Success Rate: {final_success_sampling:.1%}")
    print(f"  Mean Episode Length: {final_length_sampling:.1f}")
    
    # Evaluate with argmax
    final_score_argmax, final_success_argmax, final_length_argmax = evaluate_policy(
        agent, eval_env, n_eval=200, device=device, argmax=True
    )
    print(f"\nFinal Evaluation (Argmax/Greedy):")
    print(f"  Mean Score: {final_score_argmax:.2f}")
    print(f"  Success Rate: {final_success_argmax:.1%}")
    print(f"  Mean Episode Length: {final_length_argmax:.1f}")
    
    # Log to TensorBoard
    writer.add_scalar("eval/final_score_sampling", final_score_sampling, global_step)
    writer.add_scalar("eval/final_score_argmax", final_score_argmax, global_step)
    writer.add_scalar("eval/final_success_sampling", final_success_sampling, global_step)
    writer.add_scalar("eval/final_success_argmax", final_success_argmax, global_step)
    
    # Save final model
    final_model_path = f"runs/{run_name}/final_agent.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(agent.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    envs.close()
    writer.close()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to runs/{run_name}/")
    print(f"{'='*80}\n")
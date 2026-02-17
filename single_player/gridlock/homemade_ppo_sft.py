"""
PPO with Value Network - FINAL VERSION

Fixes applied:
1. Hybrid action masking: 90% masked, 10% exploration with penalties
2. Simple sparse rewards: all reward at end, customizable via config
3. Clean structure: SFT actor → pretrain critic → PPO
4. All previous fixes: value clipping, terminal states, early stopping, etc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
from typing import Tuple, List, Optional, Dict
import time
import pickle
import os


# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

State = namedtuple('State', ('grid', 'num'))
Action = namedtuple('Action', ['idx'])
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'reward'])
Episode = namedtuple('Episode', ['states', 'actions', 'log_probs', 'values', 'rewards', 'final_score', 'valid_masks'])


class Square:
    def __init__(self, row_or_idx, col=None):
        if col is None:
            self.row = row_or_idx // 3
            self.col = row_or_idx % 3
        else:
            self.row = row_or_idx
            self.col = col
    
    def validate(self) -> bool:
        return (0 <= self.row < 3) and (0 <= self.col < 3)

    def left(self) -> "Square":
        return Square(self.row, self.col - 1)

    def right(self) -> "Square":
        return Square(self.row, self.col + 1)
    
    def up(self) -> "Square":
        return Square(self.row - 1, self.col)
    
    def down(self) -> "Square":
        return Square(self.row + 1, self.col)


def validate_action(grid: np.ndarray, num: int, square: Square) -> bool:
    """Check if placing num at square is valid"""
    if not square.validate() or grid[square.row, square.col] != 0:
        return False
    
    left = square.left()
    if left.validate() and grid[left.row, left.col] != 0:
        if grid[left.row, left.col] >= num:
            return False

    right = square.right()
    if right.validate() and grid[right.row, right.col] != 0:
        if grid[right.row, right.col] <= num:
            return False
    
    up = square.up()
    if up.validate() and grid[up.row, up.col] != 0:
        if grid[up.row, up.col] <= num:
            return False
    
    down = square.down()
    if down.validate() and grid[down.row, down.col] != 0:
        if grid[down.row, down.col] >= num:
            return False
    
    return True

def no_valid_moves(grid: np.ndarray, num: int) -> bool:
    """Check if there are any valid moves for a given card"""
    for i in range(3):
        for j in range(3):
            if validate_action(grid, num, Square(i, j)):
                return False  
    return True

def score(grid: np.ndarray) -> int:
    """Calculate score from completed rows, columns, and diagonals"""
    total = 0

    # Complete columns
    col_complete = (grid != 0).all(axis=0)
    total += (grid * col_complete).sum()

    # Complete rows
    row_complete = (grid != 0).all(axis=1)
    total += (grid * row_complete[:, None]).sum()

    # Main diagonal
    main_diag = np.diag(grid)
    if (main_diag != 0).all():
        total += main_diag.sum()

    # Anti-diagonal
    anti_diag = np.diag(np.fliplr(grid))
    if (anti_diag != 0).all():
        total += anti_diag.sum()

    return int(total)


def sample_draw_batch(batch_size: int) -> np.ndarray:
    """Sample a batch of random card sequences"""
    base = np.repeat(np.arange(1, 11), 4) # np.repeat(np.arange(1, 10), 1) -> simple
    batch = np.empty((batch_size, base.size), dtype=np.int32)
    
    for i in range(batch_size):
        batch[i] = np.random.permutation(base)
    
    return batch


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone with two heads:
    - Actor: policy (action probabilities)
    - Critic: value function (state value estimate)
    """
    def __init__(self, hidden_size=128):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(10, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        
        # Actor head
        self.actor = nn.Linear(64, 9)
        
        # Critic head with proper initialization
        self.critic = nn.Linear(64, 1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)  # Start predicting ~25
        
    def forward(self, state_encoding):
        """Returns: (action_logits, state_value)"""
        x = F.relu(self.fc1(state_encoding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        action_logits = self.actor(x)
        state_value = self.critic(x).squeeze(-1)
        
        return action_logits, state_value


class PPOPolicy:
    """PPO policy with value network"""
    def __init__(self, hidden_size=128, learning_rate=3e-4):
        self.network = ActorCriticNetwork(hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
    def encode_state(self, grid: np.ndarray, num: int) -> torch.Tensor:
        """Encode state as tensor"""
        grid_flat = grid.flatten() / 10.0
        num_normalized = num / 10.0
        state_vector = np.append(grid_flat, num_normalized)
        return torch.FloatTensor(state_vector).to(device)
    
    def encode_state_batch(self, grids: np.ndarray, nums: np.ndarray) -> torch.Tensor:
        """Batch encode states"""
        batch_size = grids.shape[0]
        grids_flat = grids.reshape(batch_size, -1) / 10.0
        nums_normalized = nums.reshape(-1, 1) / 10.0
        state_vectors = np.concatenate([grids_flat, nums_normalized], axis=1)
        return torch.FloatTensor(state_vectors).to(device)
    
    def get_action_and_value(self, state_encoding: torch.Tensor):
        """Get action distribution and value estimate"""
        return self.network(state_encoding)
    
    def get_valid_action_mask(self, grid: np.ndarray, num: int) -> np.ndarray:
        """Get mask of valid actions (1 = valid, 0 = invalid)"""
        mask = np.zeros(9, dtype=np.float32)
        for idx in range(9):
            square = Square(idx)
            if validate_action(grid, num, square):
                mask[idx] = 1.0
        
        # comment out (?)
        if mask.sum() == 0:
            mask = np.ones(9, dtype=np.float32)
        
        return mask
    
    def sample_action(self, state: State, return_log_prob=False, exploration_rate=0.0, argmax=False):
        """
        Sample action with optional exploration of invalid moves.
        
        Args:
            state: Current game state
            return_log_prob: Whether to return log probability
            exploration_rate: Probability of sampling without masking (allows invalid actions)
        """
        state_encoding = self.encode_state(state.grid, state.num)
        logits, value = self.get_action_and_value(state_encoding.unsqueeze(0))
        
        # FIX 1: Hybrid masking with exploration
        # Most of the time (90%), mask invalid actions
        # Sometimes (10%), allow sampling invalid actions to learn they're bad
        should_explore = (np.random.random() < exploration_rate)
        
        if should_explore:
            # No masking - can sample invalid actions
            masked_logits = logits
        else:
            # Mask invalid actions
            valid_mask = self.get_valid_action_mask(state.grid, state.num)
            valid_mask_tensor = torch.FloatTensor(valid_mask).to(device)
            masked_logits = logits.clone()
            masked_logits[0, valid_mask_tensor == 0] = float('-inf')

        dist = Categorical(logits=masked_logits)

        if argmax:
            action_idx = masked_logits.argmax(dim=1)
        else:
            action_idx = dist.sample()
        
        if return_log_prob:
            log_prob = dist.log_prob(action_idx)
            return Action(action_idx.item()), log_prob, value.item()
        return Action(action_idx.item())
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        """Load model"""
        self.network.load_state_dict(torch.load(path, map_location=device))


def collect_episode(
    policy: PPOPolicy, 
    draw: np.ndarray, 
    reward_config: Optional[Dict] = None,
    exploration_rate: float = 0.1
) -> Episode:
    """
    Collect a single episode with customizable reward scheme.
    
    Args:
        policy: Policy to use
        draw: Card sequence
        reward_config: Reward configuration dict
        exploration_rate: Probability of unmasked sampling
    
    FIX 2: Simple, customizable reward allocation
    """
    if reward_config is None:
        reward_config = {
            'scheme': 'sparse',
            'invalid_penalty': -50.0,
            'early_termination_penalty': 0.0,
            'step_bonus': 0.0
        }
    
    grid = np.zeros((3, 3), dtype=np.int32)
    
    states = []
    actions = []
    log_probs = []
    values = []
    valid_masks = []
    
    invalid_action_taken = False
    num_cards_placed = 0
    
    for turn in range(min(9, len(draw))):
        if no_valid_moves(grid, draw[turn]):
            break

        state = State(grid.copy(), int(draw[turn]))
        valid_mask = policy.get_valid_action_mask(grid, draw[turn])
        
        # Sample action (with exploration)
        action, log_prob, value = policy.sample_action(
            state, 
            return_log_prob=True,
            exploration_rate=exploration_rate
        )
        square = Square(action.idx)
        is_valid = validate_action(grid, draw[turn], square)
        
        # Store experience
        states.append(state)
        actions.append(action.idx)
        log_probs.append(log_prob)
        values.append(value)
        valid_masks.append(valid_mask)
        
        if not is_valid:
            invalid_action_taken = True
            break
        
        # Place card
        grid[square.row, square.col] = draw[turn]
        num_cards_placed += 1
    
    final_score = score(grid)
    terminated_early = (num_cards_placed < 9)
    
    # FIX 2: CLEAR AND SIMPLE REWARD ALLOCATION
    if reward_config['scheme'] == 'sparse':
        # All reward at the end (RECOMMENDED)
        rewards = [0.0] * len(states)
        
        if len(rewards) > 0:
            # Base reward = final score
            rewards[-1] = float(final_score)
            
            # Apply penalty for invalid action
            if invalid_action_taken:
                rewards[-1] += reward_config['invalid_penalty']
            
            # Optional: penalty for early termination
            if terminated_early and not invalid_action_taken:
                rewards[-1] += reward_config['early_termination_penalty']
            
            # Optional: bonus for each step
            if reward_config['step_bonus'] != 0:
                for i in range(len(rewards)):
                    rewards[i] += reward_config['step_bonus']
    
    elif reward_config['scheme'] == 'custom':
        # Your custom reward function here
        # This is where you can add domain knowledge
        rewards = [0.0] * len(states)
        
        # Example: bonus for completing structures
        # (You would implement your logic here)
        
        # Final score at the end
        rewards[-1] += float(final_score)
        
        if invalid_action_taken:
            rewards[-1] += reward_config['invalid_penalty']
    
    else:
        raise ValueError(f"Unknown reward scheme: {reward_config['scheme']}")
    
    return Episode(states, actions, log_probs, values, rewards, final_score, valid_masks)


def compute_gae_advantages(rewards, values, next_value, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]
        
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return advantages


def single_game_random(draw: np.ndarray, store_trajectory=False):
    """Play a game with random valid moves (for collecting expert demos)"""
    grid = np.zeros((3, 3), dtype=np.int32)
    
    if store_trajectory:
        states = []
        actions = []
    
    for i in range(min(9, len(draw))):
        state = State(grid.copy(), int(draw[i]))
        
        # Find all valid actions
        valid_actions = []
        for idx in range(9):
            square = Square(idx)
            if validate_action(grid, draw[i], square):
                valid_actions.append(idx)
        
        if len(valid_actions) == 0:
            break
        
        # Randomly choose a valid action
        action_idx = np.random.choice(valid_actions)
        square = Square(action_idx)
        
        if store_trajectory:
            states.append(state)
            actions.append(action_idx)
        
        grid[square.row, square.col] = draw[i]
    
    final_score = score(grid)
    
    if store_trajectory:
        return final_score, states, actions
    return final_score


def collect_expert_demonstrations(
    num_games: int = 10000,
    score_threshold: int = 20,
    save_path: Optional[str] = None
) -> List[Trajectory]:
    """Collect expert demonstrations by playing random games"""
    print(f"\n{'='*60}")
    print(f"Collecting Expert Demonstrations")
    print(f"{'='*60}")
    print(f"Playing {num_games} random games, filtering for score >= {score_threshold}")
    
    expert_trajectories = []
    draws = sample_draw_batch(num_games)
    
    start_time = time.time()
    scores = []
    
    for i in range(num_games):
        reward, states, actions = single_game_random(draws[i], store_trajectory=True)
        scores.append(reward)
        
        if reward >= score_threshold:
            expert_trajectories.append(Trajectory(states, actions, reward))
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{num_games} | "
                  f"Experts: {len(expert_trajectories)} | "
                  f"Avg score: {np.mean(scores[-1000:]):.2f} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"  Total games: {num_games}")
    print(f"  Expert games: {len(expert_trajectories)} ({100*len(expert_trajectories)/num_games:.1f}%)")
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Max score: {np.max(scores)}")
    print(f"  Time: {total_time:.1f}s")
    print(f"{'='*60}\n")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(expert_trajectories, f)
        print(f"Saved {len(expert_trajectories)} expert trajectories to {save_path}")
    
    return expert_trajectories


def supervised_fine_tuning_actor_only(
    expert_trajectories: List[Trajectory],
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 128
) -> PPOPolicy:
    """
    FIX 3: Train ONLY the actor via supervised learning.
    Critic will be pretrained separately.
    """
    print(f"\n{'='*60}")
    print(f"Supervised Fine-Tuning (Actor Only)")
    print(f"{'='*60}")
    print(f"Training on {len(expert_trajectories)} expert trajectories")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Extract all state-action pairs
    all_states = []
    all_actions = []
    
    for traj in expert_trajectories:
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
    
    print(f"Total state-action pairs: {len(all_states)}")
    
    # Create policy
    policy = PPOPolicy(hidden_size=hidden_size, learning_rate=learning_rate)
    
    # Convert to tensors
    state_encodings = policy.encode_state_batch(
        np.array([s.grid for s in all_states]),
        np.array([s.num for s in all_states])
    )
    actions_tensor = torch.LongTensor(all_actions).to(device)
    
    dataset_size = len(all_states)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        indices = torch.randperm(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = state_encodings[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            
            # Forward pass (only need logits for actor training)
            logits, _ = policy.get_action_and_value(batch_states)
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, batch_actions)
            
            # Backward pass
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == batch_actions).float().mean()
            epoch_accuracy += accuracy.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")
    
    print(f"{'='*60}\n")
    
    return policy


def pretrain_critic_from_policy(
    policy: PPOPolicy, 
    num_episodes: int = 500, 
    num_epochs: int = 10,
    reward_config: Optional[Dict] = None
):
    """
    FIX 3: Pretrain the critic to predict returns from policy rollouts.
    Modifies policy in-place.
    """
    print(f"\n{'='*80}")
    print(f"Pre-training Critic Network")
    print(f"{'='*80}")
    print(f"Collecting {num_episodes} episodes from current policy...")
    
    if reward_config is None:
        reward_config = {
            'scheme': 'sparse',
            'invalid_penalty': -20.0,
            'early_termination_penalty': 0.0,
            'step_bonus': 0.0
        }
    
    # Collect episodes
    draws = sample_draw_batch(num_episodes)
    all_states = []
    all_returns = []
    
    for i in range(num_episodes):
        episode = collect_episode(policy, draws[i], reward_config=reward_config, exploration_rate=0.1)  # changed exploration_rate from 0.0
        if len(episode.states) == 0:
            continue
        
        # Compute discounted returns
        returns = []
        G = 0.0
        gamma = 0.99
        for r in reversed(episode.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        all_states.extend(episode.states)
        all_returns.extend(returns)
    
    # Convert to tensors
    state_encodings = policy.encode_state_batch(
        np.array([s.grid for s in all_states]),
        np.array([s.num for s in all_states])
    )
    returns_tensor = torch.FloatTensor(all_returns).to(device)
    
    print(f"Collected {len(all_states)} state-return pairs")
    print(f"Training critic for {num_epochs} epochs...")
    
    # Train only the critic
    critic_optimizer = torch.optim.Adam(
        list(policy.network.critic.parameters()),
        lr=1e-3
    )
    
    dataset_size = len(all_states)
    batch_size = 256
    
    for epoch in range(num_epochs):
        indices = torch.randperm(dataset_size)
        epoch_loss = 0
        num_batches = 0
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = state_encodings[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            
            # Forward pass (only need values)
            _, values = policy.get_action_and_value(batch_states)
            
            # MSE loss
            loss = F.mse_loss(values, batch_returns)
            
            # Update only critic
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Check predictions
        with torch.no_grad():
            _, sample_values = policy.get_action_and_value(state_encodings[:100])
            sample_returns = returns_tensor[:100]
            mae = torch.abs(sample_values - sample_returns).mean().item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.3f} | MAE: {mae:.2f}")
    
    print(f"{'='*80}")
    print(f"Critic pre-training complete!\n")


def evaluate_policy(policy: PPOPolicy, num_games: int = 100, argmax: bool = False) -> float:
    """Evaluate policy performance"""
    draws = sample_draw_batch(num_games)
    scores = []
    
    for i in range(num_games):
        grid = np.zeros((3, 3), dtype=np.int32)
        
        for turn in range(min(9, len(draws[i]))):
            state = State(grid.copy(), int(draws[i][turn]))
            action = policy.sample_action(state, exploration_rate=0.0, argmax=argmax)  # No exploration during eval
            square = Square(action.idx)
            
            if not validate_action(grid, draws[i][turn], square):
                break
            
            grid[square.row, square.col] = draws[i][turn]
        
        scores.append(score(grid))
    
    return np.mean(scores)


def train_ppo(
    pretrained_policy: PPOPolicy,
    num_iterations: int = 500,
    episodes_per_iter: int = 128,
    num_epochs: int = 3,
    minibatch_size: int = 256,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    learning_rate: float = 1e-4,  # Lower LR for fine-tuning
    max_grad_norm: float = 0.5,
    eval_interval: int = 20,
    patience: int = 10,
    reward_config: Optional[Dict] = None,
    exploration_rate: float = 0.1,
    save_dir: str = './ckpt/ppo'
):
    """
    Train using PPO with value network.
    
    All fixes applied:
    - Hybrid action masking with exploration
    - Simple sparse rewards
    - Value clipping
    - Early stopping
    - Proper terminal state handling
    """
    
    if reward_config is None:
        reward_config = {
            'scheme': 'sparse',
            'invalid_penalty': -20.0,
            'early_termination_penalty': 0.0,
            'step_bonus': 0.0
        }
    
    policy = pretrained_policy
    
    best_score = 0
    iterations_without_improvement = 0
    
    print(f"\n{'='*80}")
    print(f"PPO Training")
    print(f"{'='*80}")
    print(f"Episodes per iteration: {episodes_per_iter}")
    print(f"Exploration rate: {exploration_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Reward config: {reward_config}")
    print(f"Early stopping patience: {patience}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Collect episodes
        draws = sample_draw_batch(episodes_per_iter)
        episodes = []
        
        for i in range(episodes_per_iter):
            episode = collect_episode(
                policy, 
                draws[i], 
                reward_config=reward_config,
                exploration_rate=exploration_rate
            )
            if len(episode.states) > 0:
                episodes.append(episode)
        
        if len(episodes) == 0:
            print(f"Iteration {iteration}: No episodes collected!")
            continue
        
        # Flatten all episode data
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_old_values = []
        all_advantages = []
        all_returns = []
        all_valid_masks = []
        
        for episode in episodes:
            next_value = 0.0
            advantages = compute_gae_advantages(
                episode.rewards,
                episode.values,
                next_value,
                gamma,
                gae_lambda
            )
            
            returns = [adv + val for adv, val in zip(advantages, episode.values)]
            
            all_states.extend(episode.states)
            all_actions.extend(episode.actions)
            all_old_log_probs.extend(episode.log_probs)
            all_old_values.extend(episode.values)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_valid_masks.extend(episode.valid_masks)
        
        # Convert to tensors
        state_encodings = policy.encode_state_batch(
            np.array([s.grid for s in all_states]),
            np.array([s.num for s in all_states])
        )
        actions_tensor = torch.LongTensor(all_actions).to(device)
        old_log_probs = torch.stack(all_old_log_probs).detach()
        old_values_tensor = torch.FloatTensor(all_old_values).to(device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(device)
        returns_tensor = torch.FloatTensor(all_returns).to(device)
        valid_masks_tensor = torch.FloatTensor(np.array(all_valid_masks)).to(device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        dataset_size = len(all_states)
        
        # PPO update
        for epoch in range(num_epochs):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = state_encodings[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_valid_masks = valid_masks_tensor[batch_indices]
                
                # Forward pass
                logits, values = policy.get_action_and_value(batch_states)
                
                # Apply action masking
                masked_logits = logits.clone()
                masked_logits[batch_valid_masks == 0] = float('-inf')
                
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Value loss with clipping
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -value_clip_epsilon,
                    value_clip_epsilon
                )
                value_loss_unclipped = F.mse_loss(values, batch_returns, reduction='none')
                value_loss_clipped = F.mse_loss(value_pred_clipped, batch_returns, reduction='none')
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                # Update
                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.network.parameters(), max_grad_norm)
                policy.optimizer.step()
        
        # Logging
        final_scores = [ep.final_score for ep in episodes]
        mean_score = np.mean(final_scores)
        max_score = np.max(final_scores)
        avg_episode_length = np.mean([len(ep.states) for ep in episodes])
        avg_value = np.mean([np.mean(ep.values) for ep in episodes])
        iter_time = time.time() - iter_start
        
        print(f"Iter {iteration:3d} | Mean: {mean_score:6.2f} | Max: {max_score:3.0f} | "
              f"AvgLen: {avg_episode_length:.1f} | AvgV: {avg_value:.1f} | Time: {iter_time:.2f}s")
        
        # Evaluation
        if (iteration + 1) % eval_interval == 0 or iteration == 0:
            eval_score = evaluate_policy(policy, num_games=200)
            print(f"  >>> Evaluation: {eval_score:.2f} (Best: {best_score:.2f})")
            
            if eval_score > best_score:
                best_score = eval_score
                iterations_without_improvement = 0
                policy.save(f'{save_dir}/best_ppo_policy.pt')
                print(f"  >>> NEW BEST! Saved.")
            else:
                iterations_without_improvement += 1
                print(f"  >>> No improvement ({iterations_without_improvement}/{patience})")
            
            # Early stopping
            if iterations_without_improvement >= patience:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING at iteration {iteration}")
                print(f"No improvement for {patience} evaluations")
                print(f"Best score: {best_score:.2f}")
                print(f"{'='*80}\n")
                policy.load(f'{save_dir}/best_ppo_policy.pt')
                break
    
    print(f"\n{'='*80}")
    print(f"Training complete! Best score: {best_score:.2f}")
    print(f"{'='*80}\n")
    
    return policy


if __name__ == "__main__":
    # Configuration
    SAVE_DIR = './ckpt/ppo_sft_final'
    
    # FIX 2: Simple, clear reward configuration
    REWARD_CONFIG = {
        'scheme': 'sparse',              # All reward at end
        'invalid_penalty': -20.0,        # Strong penalty for invalid moves
        'early_termination_penalty': 0.0,  # Let value network learn if early stop is good/bad
        'step_bonus': 0.0                # No bonus for survival
    }
    
    EXPLORATION_RATE = 0.1  # 10% chance of unmasked sampling
    
    print("="*80)
    print("FULL TRAINING PIPELINE: EXPERT COLLECTION → SFT → CRITIC PRETRAIN → PPO")
    print("="*80)
    
    # Step 1: Collect expert demonstrations
    expert_trajs = collect_expert_demonstrations(
        num_games=50000,
        score_threshold=20,
        save_path=f'{SAVE_DIR}/expert_demos.pkl'
    )
    
    if len(expert_trajs) == 0:
        print("No expert demonstrations collected! Cannot proceed.")
        exit(1)
    
    # FIX 3: Step 2 - SFT (actor only)
    print("\n" + "="*80)
    print("STEP 2: SUPERVISED FINE-TUNING (ACTOR ONLY)")
    print("="*80)
    sft_policy = supervised_fine_tuning_actor_only(
        expert_trajs,
        num_epochs=10,
        batch_size=256,
        learning_rate=1e-3
    )
    
    # Evaluate SFT actor
    sft_score = evaluate_policy(sft_policy, num_games=200)
    print(f"\nSFT Actor Evaluation: {sft_score:.2f}")
    sft_policy.save(f'{SAVE_DIR}/sft_actor_only.pt')
    
    # FIX 3: Step 3 - Pretrain critic
    print("\n" + "="*80)
    print("STEP 3: PRETRAIN CRITIC FROM SFT ROLLOUTS")
    print("="*80)
    pretrain_critic_from_policy(
        sft_policy, 
        num_episodes=500, 
        num_epochs=10,
        reward_config=REWARD_CONFIG
    )
    
    # Evaluate SFT with critic
    sft_with_critic_score = evaluate_policy(sft_policy, num_games=200)
    print(f"\nSFT with Critic Evaluation: {sft_with_critic_score:.2f}")
    sft_policy.save(f'{SAVE_DIR}/sft_with_critic.pt')
    
    # Step 4: PPO training
    print("\n" + "="*80)
    print("STEP 4: PPO TRAINING")
    print("="*80)
    trained_policy = train_ppo(
        pretrained_policy=sft_policy,
        num_iterations=1000,
        episodes_per_iter=256,
        learning_rate=1e-4,  # Reduced for fine-tuning
        reward_config=REWARD_CONFIG,
        exploration_rate=EXPLORATION_RATE,
        patience=10,
        save_dir=SAVE_DIR
    )
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    final_softmax_score = evaluate_policy(trained_policy, num_games=500)
    print(f"Final average score over 500 games with softmax sampling: {final_softmax_score:.2f}")
    final_argmax_score = evaluate_policy(trained_policy, num_games=500, argmax=True)
    print(f"Final average score over 500 games with argmax sampling: {final_argmax_score:.2f}")
    
    trained_policy.save(f'{SAVE_DIR}/final_policy.pt')
    print(f"\nAll models saved to {SAVE_DIR}/")
    print("\nTraining complete! 🎉")
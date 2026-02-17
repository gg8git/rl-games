"""
Connect Four — Gumbel MuZero with Self-Play (MPS/CPU Compatible)
================================================================
"""

import sys
import torch
import numpy as np
import gymnasium as gym
from easydict import EasyDict
from typing import Optional, Dict, Any, List

# --- LightZero & Ding Imports ---
from lzero.entry import train_muzero
from ding.utils import set_pkg_seed, ENV_REGISTRY
from ding.envs import BaseEnv, BaseEnvTimestep

# --- PettingZoo Import ---
from pettingzoo.classic import connect_four_v3

# =============================================================================
# 1. ROBUST ENVIRONMENT WRAPPER
# =============================================================================

@ENV_REGISTRY.register('connect_four')
class ConnectFourEnv(gym.Env):
    """
    Adapts PettingZoo Connect4 (AEC) to LightZero (Gym+Dict).
    """
    # LightZero expects these class attributes sometimes
    config = dict()

    def __init__(self, cfg: dict = None):
        self.cfg = cfg
        # render_mode=None is faster for training
        self._env = connect_four_v3.env(render_mode=None)
        self._agents = ['player_0', 'player_1']
        self._agent_map = {'player_0': 0, 'player_1': 1}
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, 6, 7), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(7)
        self.reward_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self._seed = 0
        self._current_step = 0

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """LightZero calls this to seed the env."""
        self._seed = seed

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict:
        """
        Resets environment and returns the observation dict for the FIRST player.
        """
        if seed is not None:
            self._seed = seed
            
        self._env.reset(seed=self._seed)
        self._current_step = 0
        
        # Get state for the first agent
        obs, _, _, _, _ = self._env.last()
        current_agent = self._env.agent_selection
        
        return self._make_obs_dict(obs, current_agent)

    def step(self, action: int):
        """
        Execute action.
        CRITICAL: LightZero expects the reward for the agent who JUST moved.
        """
        # 1. The agent who is moving NOW
        acting_agent = self._env.agent_selection
        
        # 2. Execute action
        self._env.step(action)
        
        # 3. Get the state of the NEXT agent (or same agent if game over)
        next_obs, _, term, trunc, info = self._env.last()
        done = term or trunc
        self._current_step += 1

        # 4. Extract reward for the `acting_agent`
        # PettingZoo rewards dict is populated *after* the move.
        raw_reward = self._env.rewards[acting_agent]
        
        # 5. Who plays next?
        next_agent = self._env.agent_selection
        
        obs_dict = self._make_obs_dict(next_obs, next_agent)
        
        # 6. Returns: (obs, reward, done, info)
        # Note: We must cast reward to float for LightZero
        return BaseEnvTimestep(
            obs_dict,
            float(raw_reward),
            done,
            info
        )

    def _make_obs_dict(self, pz_obs, current_agent_str):
        """
        Prepares the observation dictionary.
        Shape: (3, 6, 7) -> [Current Player, Opponent, ToPlay_indicator]
        """
        # pz_obs['observation'] is (6, 7, 2) -> (Rows, Cols, Channels)
        # Channel 0: Current Player pieces
        # Channel 1: Opponent pieces
        board = pz_obs['observation']
        
        # Transpose to (2, 6, 7)
        board_planes = np.transpose(board, (2, 0, 1)).astype(np.float32)
        
        # Feature 3: To-play plane (Optional but helps some models)
        # Not strictly required for Gumbel, but good practice.
        # We will stick to the 2 planes you used, but LightZero Gumbel 
        # often defaults to expecting 3 channels (C, H, W) or specific config.
        # Let's stick to your 2 planes to keep it simple, but ensure shape is right.
        
        to_play_int = np.array([self._agent_map[current_agent_str]], dtype=np.int64)
        
        return {
            'observation': board_planes,
            'action_mask': pz_obs['action_mask'].astype(np.float32),
            'to_play': to_play_int,
            'timestep': self._current_step
        }

    def close(self):
        self._env.close()

    @classmethod
    def create_collector_env_cfg(cls, cfg):
        # Helper required by LightZero's manager
        collector_env_num = cfg.pop('collector_env_num')
        return [cfg for _ in range(collector_env_num)]

    @classmethod
    def create_evaluator_env_cfg(cls, cfg):
        # Helper required by LightZero's manager
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [cfg for _ in range(evaluator_env_num)]


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

def get_config():
    # Force CPU by default for stability on Mac (MPS support in LightZero is experimental)
    # If you are brave, change this to 'mps' but expect "NotImplemented" errors in TreeSearch.
    DEVICE = 'cpu' 
    
    # -------------------------------------------------------------------------
    # Gumbel MuZero Config
    # -------------------------------------------------------------------------
    main_config = dict(
        exp_name='connect4_gumbel_debug',
        env=dict(
            env_id='connect_four',
            collector_env_num=2,  # Keep low for debugging
            evaluator_env_num=2,
            n_evaluator_episode=4,
            stop_value=2.0,       # Unreachable, so it runs until max_env_step
            manager=dict(shared_memory=False), # REQUIRED for macOS
        ),
        policy=dict(
            model=dict(
                observation_shape=(2, 6, 7),
                action_space_size=7,
                image_channel=2,
                model_type='conv', 
                # Simplified Network for CPU Debugging
                num_res_blocks=1,
                num_channels=16,
                fc_reward_layers=[16],
                fc_value_layers=[16],
                fc_policy_layers=[16],
                downsample=False,
            ),
            cuda=False,  # Set False for CPU/MPS
            mcts_ctree=False, # CRITICAL: False for Python backend (Slow but works on Mac)
            
            # Search params
            num_simulations=8, # Low for fast debugging
            max_num_considered_actions=7,
            
            # Learning params
            batch_size=16, # Low for debugging
            optim_type='AdamW',
            learning_rate=0.001,
            update_per_collect=2,
            eval_freq=1_000,
            
            # Gumbel Specifics
            gumbel_algo=True,
            two_player_game=True, # Handles the minimax sign flipping
            discount_factor=1.0,
            
            # Dimensions
            game_segment_length=20,
            replay_buffer_size=1000,
        ),
    )

    create_config = dict(
        env=dict(
            type='connect_four',
            import_names=['__main__'],
        ),
        env_manager=dict(type='base'), # Simple blocking manager for debug
        policy=dict(
            type='gumbel_muzero',
            import_names=['lzero.policy.gumbel_muzero'],
        ),
    )
    
    return EasyDict(main_config), EasyDict(create_config)


# =============================================================================
# 3. DEBUG & ENTRY
# =============================================================================

def smoke_test():
    """Runs a random game to verify the env wrapper."""
    print(">>> Running Smoke Test...")
    env = ConnectFourEnv()
    env.seed(42)
    obs = env.reset()
    print(f"Observation Shape: {obs['observation'].shape}")
    print(f"Action Mask: {obs['action_mask']}")
    
    done = False
    while not done:
        # Pick a random valid move
        mask = obs['action_mask'].astype(bool)
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)
        
        obs_timestep = env.step(action)
        obs = obs_timestep.obs
        done = obs_timestep.done
        print(f"Agent acted. Reward: {obs_timestep.reward}, Done: {done}")
        
    print(">>> Smoke Test Passed.")

def train():
    """Main Training Loop."""
    main_cfg, create_cfg = get_config()
    
    # 1. Seed everything
    set_pkg_seed(0, use_cuda=False)
    
    # 2. Run LightZero Entry
    # This function handles the Loop: Collect -> Train -> Eval
    train_muzero(
        [main_cfg, create_cfg],
        seed=0,
        max_env_step=100_000, # Runs for a bit
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke_test()
    else:
        train()
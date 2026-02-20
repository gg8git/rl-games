import copy
from typing import List

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('gridlock')
class GridlockEnv(BaseEnv):
    """
    Overview:
        The GridlockEnv is a LightZero compatible environment for the single-player 
        turn-based game Gridlock. 
    """

    config = dict(
        env_id="gridlock",
        # Whether to scale the observation values to [0, 1]
        scale_obs=True,
        # (bool) Whether to use the 'channel last' format for the observation space.
        # If False, 'channel first' format is used.
        channel_last=False,
        # (int) The maximum number of steps in an episode.
        max_episode_steps=9,
        # (bool) Whether to collect data during the game.
        is_collect=True,
        # (bool) Whether to flatten the observation space. If True, the observation space is a 1D array instead of a 2D grid.
        need_flatten=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config

        self.scale_obs = self._cfg.scale_obs
        self.channel_last = self._cfg.channel_last
        self.max_episode_steps = self._cfg.max_episode_steps
        self.is_collect = self._cfg.is_collect
        self.need_flatten = self._cfg.need_flatten

        self.board_size = 3
        self.total_num_actions = self.board_size * self.board_size

        # Internal state
        self.deck = None
        self.grid = None
        self.pointer = 0
        self._current_player = 1 # Required for LightZero compatibility

        # RL Spaces
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        
        # Observation space formatted for MuZero CNNs: (2, 3, 3)
        # Channel 0: Board state
        # Channel 1: Current card to play (broadcasted)
        if self.need_flatten
            self._observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        else:
            self._observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, 3, 3), dtype=np.float32)
        
        # Max possible score is around 216
        self._reward_space = gym.spaces.Box(low=-20.0, high=250.0, shape=(1,), dtype=np.float32)

    def reset(self, start_player_index: int = 0, init_state=None) -> dict:
        self.pointer = 0
        
        # 4x 1-10 deck
        self.deck = np.tile(np.arange(1, 11), 4)
        np.random.shuffle(self.deck)
        
        if init_state is not None:
            self.grid = np.array(copy.deepcopy(init_state), dtype=np.int32)
        else:
            self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        return self.observe()

    def step(self, action: int) -> BaseEnvTimestep:
        # Validate Move
        valid_actions = self.legal_actions
        
        # Agent explicitly chose an illegal move when legal moves were available.
        # Apply penalty and terminate the episode.
        if action not in valid_actions:
            reward = -20.0
            done = True
            info = {'eval_episode_return': reward}
            return BaseEnvTimestep(self.observe(), np.array(reward, dtype=np.float32), done, info)

        # Execute valid move
        row, col = action // 3, action % 3
        self.grid[row, col] = self.deck[self.pointer]
        self.pointer += 1

        done = False
        reward = 0.0

        # Termination Conditions
        if self.pointer >= 9:
            # 1. Grid successfully filled
            done = True
            reward = self._score_grid()
        elif self.pointer < 40:
            # 2. Check if the NEXT card has any valid placements
            # If the draw is impossible to play, the game ends naturally. 
            # No penalty is applied here; the agent just keeps its current score.
            if len(self.legal_actions) == 0:
                done = True
                reward = self._score_grid()
        else:
            # Failsafe
            done = True
            reward = self._score_grid()

        info = {}
        if done:
            info['eval_episode_return'] = reward

        return BaseEnvTimestep(self.observe(), np.array(reward, dtype=np.float32), done, info)

    def observe(self) -> dict:
        # Build MuZero spatial observation
        if need_flatten:
            grid_obs = self.grid.flatten().astype(np.float32)
            card_obs = self.deck[self.pointer] if self.pointer < 40 else 0
            observation = np.append(grid_obs, card_obs)
        else:
            grid_obs = self.grid.astype(np.float32)
            
            if self.pointer < 40:
                current_card = self.deck[self.pointer]
                card_obs = np.full((3, 3), current_card, dtype=np.float32)
            else:
                card_obs = np.zeros((3, 3), dtype=np.float32)

            observation = np.stack([grid_obs, card_obs], axis=0)
        
        # scale observations
        if self.scale_obs:
                grid_obs /= 10.0
                card_obs /= 10.0

        # Build action mask
        action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
        for act in self.legal_actions:
            action_mask[act] = 1

        return {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': -1  # -1 indicates a single-player environment to MCTS
        }

    @property
    def legal_actions(self) -> List[int]:
        if self.pointer >= 9 or self.pointer >= 40:
            return []
            
        card = self.deck[self.pointer]
        valid_actions = []
        for a in range(self.total_num_actions):
            if self._validate_action(a, card):
                valid_actions.append(a)
        return valid_actions

    def _validate_action(self, action: int, card: int) -> bool:
        row, col = action // 3, action % 3
        
        if self.grid[row, col] != 0:
            return False
            
        # Left (must be < current)
        if col - 1 >= 0 and self.grid[row, col - 1] != 0:
            if self.grid[row, col - 1] >= card: 
                return False
        # Right (must be > current)
        if col + 1 < 3 and self.grid[row, col + 1] != 0:
            if self.grid[row, col + 1] <= card: 
                return False
        # Up (must be > current)
        if row - 1 >= 0 and self.grid[row - 1, col] != 0:
            if self.grid[row - 1, col] <= card: 
                return False
        # Down (must be < current)
        if row + 1 < 3 and self.grid[row + 1, col] != 0:
            if self.grid[row + 1, col] >= card: 
                return False
                
        return True

    def _score_grid(self) -> float:
        total = 0.0
        # Columns and Rows
        for i in range(3):
            if (self.grid[:, i] != 0).all(): 
                total += self.grid[:, i].sum()
            if (self.grid[i, :] != 0).all(): 
                total += self.grid[i, :].sum()
                
        # Diagonals
        if (np.diag(self.grid) != 0).all(): 
            total += np.diag(self.grid).sum()
        if (np.diag(np.fliplr(self.grid)) != 0).all(): 
            total += np.diag(np.fliplr(self.grid)).sum()
            
        return float(total)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "LightZero Gridlock Env"
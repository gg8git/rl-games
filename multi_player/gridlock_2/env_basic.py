import copy
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('gridlock2')
class Gridlock2Env(BaseEnv):
    """
    Overview:
        The Gridlock2Env is a LightZero compatible environment for the 2-3 player 
        turn-based game Gridlock2. Players draft cards in a snake order and try 
        to build the highest scoring 3x3 grid.
    """

    config = dict(
        env_id="gridlock2",
        # Supports 2 or 3 players
        num_players=2, 
        non_zero_sum=False,
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        scale_obs=True,
        channel_last=False,
        max_episode_steps=9,
        is_collect=True,
        need_flatten=False, # no support
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

        self.battle_mode = self._cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'

        self.scale_obs = self._cfg.scale_obs
        self.channel_last = self._cfg.channel_last
        self.max_episode_steps = self._cfg.max_episode_steps
        self.is_collect = self._cfg.is_collect
        self.need_flatten = self._cfg.need_flatten

        self._env = self
        self.num_players = self._cfg.num_players
        assert self.num_players in [2, 3], "Gridlock2 strictly supports 2 or 3 players."
        self.board_size = 3
        self.total_num_actions = self.board_size * self.board_size
        self.players = list(range(1, self.num_players + 1))

        # Internal state
        self.deck = None
        self.grids = None
        self.pointer = 0
        self.current_player_index = 0 # Required for LightZero compatibility
        
        # RL Spaces
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        
        # Observation space formatted for MuZero CNNs: (num_players + 1, 3, 3)
        # Ch 0: Current Player Grid
        # Ch 1 to N-1: Opponent Grids
        # Ch N: Current card to play (broadcasted)
        high = 1.0 if self.scale_obs else 10.0
        if self.need_flatten:
            self._observation_space = gym.spaces.Box(low=0.0, high=high, shape=(self.num_players,10), dtype=np.float32)
        else:
            self._observation_space = gym.spaces.Box(low=0.0, high=high, shape=(self.num_players + 1, 3, 3), dtype=np.float32)
        self._reward_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, start_player_index: int = 0, init_state=None) -> dict:
        self.pointer = 0
        
        # Shared 40-card Deck
        self.deck = np.tile(np.arange(1, 11), 4)
        np.random.shuffle(self.deck)
        
        if init_state is not None:
            self.grids = [np.array(copy.deepcopy(init_state[i]), dtype=np.int32) for i in range(self.num_players)]
        else:
            self.grids = [np.zeros((self.board_size, self.board_size), dtype=np.int32) for _ in range(self.num_players)]
        self.cards_played = [0] * self.num_players
        self.active_players = [True] * self.num_players

        # Generate a baseline Snake draft sequence up to 40 cards
        self.turn_sequence = []
        for r in range((40 // self.num_players) + 3):
            if r % 2 == 0:
                self.turn_sequence.extend(list(range(self.num_players)))
            else:
                self.turn_sequence.extend(list(reversed(range(self.num_players))))
                
        # Required to maintain LZ compatibility
        self.start_player_index = start_player_index
        self.current_player_index = self.start_player_index
        while self.turn_sequence[0] != self.start_player_index:
            self.turn_sequence.pop(0)

        # Fast-forward to the first valid turn
        self._advance_turn()
        return self.observe()

    def _advance_turn(self):
        """
        Pops from the turn sequence until it finds an active player with legal moves.
        If a player has no legal moves for the drawn card, they are eliminated and the card is burned.
        """
        while self.pointer < 40 and any(self.active_players) and self.turn_sequence:
            p = self.turn_sequence[0]
            
            # Skip if they're already inactive or their grid is full
            if not self.active_players[p] or self.cards_played[p] >= 9:
                self.active_players[p] = False
                self.turn_sequence.pop(0)
                continue
                
            self.current_player_index = p
            
            # If no legal actions, player is eliminated and the card is discarded
            if len(self.legal_actions) == 0:
                self.active_players[p] = False
                self.turn_sequence.pop(0)
                self.pointer += 1
                continue
                
            break  # Valid turn found

    def _player_step(self, action: int) -> BaseEnvTimestep:
        acting_player = self.current_player_index

        # Execute valid move or penalize with random action
        row, col = action // 3, action % 3
        if action in self.legal_actions:
            self.grids[acting_player][row, col] = self.deck[self.pointer]
            self.cards_played[acting_player] += 1
        else:
            self.active_players[acting_player] = False
            
        self.turn_sequence.pop(0)
        self.pointer += 1
        
        # Proceed to the next player's turn
        self._advance_turn()
        
        done, winner = self.get_done_winner()
        reward = 0.0
        info = {}
        
        if done:
            # Reward assignment for the player who *just* moved
            if winner == self.players[acting_player]:
                reward = 1.0
            elif winner == -1:
                reward = 0.0
            else:
                reward = -1.0
                
            # LightZero evaluates the episode return primarily from Player 1's perspective
            winners = get_winners() 
            if 0 in winners and len(winners) == 1:
                info['eval_episode_return'] = 1.0
            elif 0 in winners:
                info['eval_episode_return'] = 0.0
            else:
                info['eval_episode_return'] = -1.0
                
        return BaseEnvTimestep(self.observe(), np.array(reward, dtype=np.float32), done, info)

    def step(self, action: int) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            return self._player_step(action)
        else:
            raise NotImplementedError("Only self_play_mode is configured for Gridlock2.")

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # Sort grids relative to the current player's perspective
        grids_obs = [self.grids[self.current_player_index].astype(np.float32)]
        for i in range(1, self.num_players):
            opp_idx = (self.current_player_index + i) % self.num_players
            grids_obs.append(self.grids[opp_idx].astype(np.float32))
            
        # Append the broadcasted card (or zeros if the game is over)
        if self.pointer < 40 and any(self.active_players) and len(self.turn_sequence) > 0:
            current_card = self.deck[self.pointer]
            card_obs = np.full((3, 3), current_card, dtype=np.float32)
        else:
            card_obs = np.zeros((3, 3), dtype=np.float32)
            
        raw_obs = np.stack(grids_obs + [card_obs], axis=0)
        scale_obs = raw_obs / 10.0 if self.scale_obs else raw_obs
        
        if self.channel_last:
            return np.transpose(raw_obs, [1, 2, 0]), np.transpose(scale_obs, [1, 2, 0])
        else:
            return raw_obs, scale_obs

    def observe(self) -> dict:
        action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
        for act in self.legal_actions:
            action_mask[act] = 1

        return {
            'observation': self.current_state()[1],
            'action_mask': action_mask,
            'to_play': self._current_player if self.battle_mode == 'self_play_mode' else -1
        }

    @property
    def legal_actions(self) -> List[int]:
        if self.pointer >= 40:
            return []
            
        card = self.deck[self.pointer]
        grid = self.grids[self.current_player_index]
        valid_actions = []
        for a in range(self.total_num_actions):
            if self._validate_action(a, card, grid):
                valid_actions.append(a)
        return valid_actions

    def _validate_action(self, action: int, card: int, grid: np.ndarray) -> bool:
        row, col = action // 3, action % 3
        
        if grid[row, col] != 0:
            return False
            
        # Left (must be < current)
        if col - 1 >= 0 and grid[row, col - 1] != 0 and grid[row, col - 1] >= card: 
            return False
        # Right (must be > current)
        if col + 1 < 3 and grid[row, col + 1] != 0 and grid[row, col + 1] <= card: 
            return False
        # Up (must be > current)
        if row - 1 >= 0 and grid[row - 1, col] != 0 and grid[row - 1, col] <= card: 
            return False
        # Down (must be < current)
        if row + 1 < 3 and grid[row + 1, col] != 0 and grid[row + 1, col] >= card: 
            return False
                
        return True

    def _score_grid(self, grid: np.ndarray) -> float:
        total = 0.0
        for i in range(3):
            if (grid[:, i] != 0).all(): total += grid[:, i].sum()
            if (grid[i, :] != 0).all(): total += grid[i, :].sum()
                
        if (np.diag(grid) != 0).all(): total += np.diag(grid).sum()
        if (np.diag(np.fliplr(grid)) != 0).all(): total += np.diag(np.fliplr(grid)).sum()
            
        return float(total)

    def get_done_winner(self) -> Tuple[bool, int]:
        if self.pointer >= 40 or not any(self.active_players) or len(self.turn_sequence) == 0:
            winners = get_winners()
            if len(winners) == 1:
                return True, self.players[winners[0]]
            else:
                return True, -1 # Draw
        return False, -1
    
    def get_winners(self) -> List[int]:
        scores = [self._score_grid(g) for g in self.grids]
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]
        return winners
    
    @property
    def _current_player(self):
        return self.players[self.current_player_index]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return "LightZero Gridlock2 Env"
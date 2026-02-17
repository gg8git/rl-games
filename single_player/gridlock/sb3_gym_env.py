"""
gym_env_sb3.py — Gridlock environment, SB3-compatible.

The only meaningful change from gym_env_final.py is the addition of
`action_masks()`, which is the interface expected by sb3-contrib's
MaskablePPO / ActionMasker wrapper.  Everything else is identical so
the two files stay in sync and you can swap them freely.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Game Logic Helpers
# ---------------------------------------------------------------------------

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

    def left(self):  return Square(self.row, self.col - 1)
    def right(self): return Square(self.row, self.col + 1)
    def up(self):    return Square(self.row - 1, self.col)
    def down(self):  return Square(self.row + 1, self.col)


def validate_action(grid: np.ndarray, num: int, square: Square) -> bool:
    if not square.validate() or grid[square.row, square.col] != 0:
        return False

    neighbors = [
        (square.left(),  'ge'),
        (square.right(), 'le'),
        (square.up(),    'le'),
        (square.down(),  'ge'),
    ]
    for neighbor, op in neighbors:
        if neighbor.validate() and grid[neighbor.row, neighbor.col] != 0:
            val = grid[neighbor.row, neighbor.col]
            if op == 'ge' and val >= num: return False
            if op == 'le' and val <= num: return False
    return True


def score_grid(grid: np.ndarray) -> float:
    total = 0.0
    for i in range(3):
        if (grid[:, i] != 0).all(): total += grid[:, i].sum()   # cols
        if (grid[i, :] != 0).all(): total += grid[i, :].sum()   # rows
    if (np.diag(grid) != 0).all():            total += np.diag(grid).sum()
    if (np.diag(np.fliplr(grid)) != 0).all(): total += np.diag(np.fliplr(grid)).sum()
    return float(total)


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class GridlockEnv(gym.Env):
    """
    Gridlock card-placement game.

    Observation : Box(10,)  — 9 grid cells + current card, all normalised to [0, 1]
    Action      : Discrete(9) — place card in one of 9 grid positions

    The `action_masks()` method is consumed by sb3-contrib's MaskablePPO.
    The legacy `get_action_mask()` is kept for backwards-compatibility with
    any custom evaluation / data-collection code.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.deck: np.ndarray | None = None
        self.grid: np.ndarray | None = None
        self.pointer: int = 0

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = np.tile(np.arange(1, 11), 4)
        np.random.shuffle(self.deck)
        self.grid = np.zeros((3, 3), dtype=np.int32)
        self.pointer = 0
        return self._get_obs(), {"action_mask": self.get_action_mask()}

    def step(self, action):
        current_card = self.deck[self.pointer]
        sq = Square(action)

        if not validate_action(self.grid, current_card, sq):
            # Should never happen when masks are respected; kept as a failsafe.
            return self._get_obs(), -1.0, True, False, {"action_mask": self.get_action_mask()}

        self.grid[sq.row, sq.col] = current_card
        self.pointer += 1

        terminated = False
        reward = 0.0

        if self.pointer >= 9:
            terminated = True
            reward = score_grid(self.grid)
        elif self.pointer < 40:
            next_card = self.deck[self.pointer]
            if self._compute_mask(self.grid, next_card).sum() == 0:
                terminated = True
                reward = score_grid(self.grid)

        return self._get_obs(), reward, terminated, False, {"action_mask": self.get_action_mask()}

    # ------------------------------------------------------------------
    # Action masking — two equivalent interfaces
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        SB3 / sb3-contrib interface.
        Called automatically by MaskablePPO at every step.
        """
        return self.get_action_mask()

    def get_action_mask(self) -> np.ndarray:
        """Legacy interface used by data-collection / evaluation helpers."""
        if self.pointer >= 9 or self.pointer >= 40:
            return np.zeros(9, dtype=bool)
        return self._compute_mask(self.grid, self.deck[self.pointer])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        grid_flat   = self.grid.flatten() / 10.0
        current_card = self.deck[self.pointer] / 10.0 if self.pointer < 40 else 0.0
        return np.append(grid_flat, current_card).astype(np.float32)

    def _compute_mask(self, grid: np.ndarray, card: int) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        for i in range(9):
            if validate_action(grid, card, Square(i)):
                mask[i] = True
        return mask
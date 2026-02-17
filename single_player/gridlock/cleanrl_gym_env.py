import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- Game Logic Helpers (Copied & Adapted from your script) ---
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

    def left(self): return Square(self.row, self.col - 1)
    def right(self): return Square(self.row, self.col + 1)
    def up(self): return Square(self.row - 1, self.col)
    def down(self): return Square(self.row + 1, self.col)

def validate_action(grid: np.ndarray, num: int, square: Square) -> bool:
    if not square.validate() or grid[square.row, square.col] != 0:
        return False
    
    # Check neighbors
    neighbors = [
        (square.left(), 'ge'),   # left must be < current (current > left)
        (square.right(), 'le'),  # right must be > current (current < right)
        (square.up(), 'le'),     # up must be > current
        (square.down(), 'ge')    # down must be < current
    ]
    
    for neighbor, op in neighbors:
        if neighbor.validate() and grid[neighbor.row, neighbor.col] != 0:
            val = grid[neighbor.row, neighbor.col]
            if op == 'ge' and val >= num: return False
            if op == 'le' and val <= num: return False
    return True

def score_grid(grid: np.ndarray) -> float:
    total = 0
    # Cols, Rows, Diagonals logic
    for i in range(3):
        if (grid[:, i] != 0).all(): total += grid[:, i].sum() # Cols
        if (grid[i, :] != 0).all(): total += grid[i, :].sum() # Rows
        
    if (np.diag(grid) != 0).all(): total += np.diag(grid).sum() # Main diag
    if (np.diag(np.fliplr(grid)) != 0).all(): total += np.diag(np.fliplr(grid)).sum() # Anti diag
    return float(total)

# --- Gymnasium Environment ---
class GridlockEnv(gym.Env):
    """
    Gridlock Environment
    Observation: Box(10,) -> [9 grid cells normalized, 1 current card normalized]
    Action: Discrete(9) -> Place card in one of 9 spots
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.deck = None
        self.grid = None
        self.pointer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 4x 1-10 deck
        self.deck = np.tile(np.arange(1, 11), 4)  # np.tile(np.arange(1, 10), 1) -> simple
        np.random.shuffle(self.deck)
        # self.deck = np.append(self.deck, 0)  # -> simple
        
        self.grid = np.zeros((3, 3), dtype=np.int32)
        self.pointer = 0
        return self._get_obs(), {"action_mask": self.get_action_mask()}

    def step(self, action):
        current_card = self.deck[self.pointer]
        sq = Square(action)
        
        # Validate Move
        if not validate_action(self.grid, current_card, sq):
            # Invalid move penalty (should be handled by masking, but failsafe here)
            return self._get_obs(), -1.0, True, False, {"action_mask": self.get_action_mask()}

        # Place card
        self.grid[sq.row, sq.col] = current_card
        self.pointer += 1
        
        # Check termination
        # 1. Grid full (9 moves)
        # 2. No valid moves for NEXT card
        terminated = False
        reward = 0.0
        
        if self.pointer >= 9:
            terminated = True
            reward = score_grid(self.grid)
        elif self.pointer < 40:
             # Check if next card has ANY valid moves
            next_card = self.deck[self.pointer]
            mask = self._compute_mask(self.grid, next_card)
            if mask.sum() == 0:
                terminated = True
                reward = score_grid(self.grid)

        return self._get_obs(), reward, terminated, False, {"action_mask": self.get_action_mask()}

    def _get_obs(self):
        # Normalize to 0-1 range as requested in original script
        grid_flat = self.grid.flatten() / 10.0
        current_card = self.deck[self.pointer] / 10.0 if self.pointer < 40 else 0
        return np.append(grid_flat, current_card).astype(np.float32)

    def get_action_mask(self):
        # Safety: If game is over or pointer invalid, return all False (or handle gracefully)
        if self.pointer >= 9 or self.pointer >= 40:
            return np.zeros(9, dtype=bool)
        
        card = self.deck[self.pointer]
        mask = self._compute_mask(self.grid, card)
        
        # SAFETY FIX: If the mask is empty, we are in a deadlock.
        # We cannot fix it here (since this is a query), but we return the empty mask.
        # The Agent's new fallback logic (above) will handle this by forcing a bad move
        # which triggers the step() invalid move penalty.
        return mask
    
    def _compute_mask(self, grid, card):
        mask = np.zeros(9, dtype=bool)
        for i in range(9):
            if validate_action(grid, card, Square(i)):
                mask[i] = True
        return mask
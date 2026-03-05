"""
base_env.py
──────────────
A self-contained GongZhu (宫主) environment for 4 players, designed to generate
training data for the GongZhuBeliefPredictor model.

Card encoding
─────────────
  card_index = suit * 13 + (rank - 2)

  Suits  : Clubs=0  Diamonds=1  Hearts=2  Spades=3
  Ranks  : 2→0  3→1  …  10→8  J→9  Q→10  K→11  A→12

Special cards
─────────────
  Q♠  = 49   -200 pts
  J♦  = 22   +100 pts
  10♣ =  8   doubles total score (or +50 if no other scoring cards)
  2♥–4♥ = 26–28   0 pts individual (still count toward grand-slam)
  5♥–10♥ = 29–34  -10 pts each
  J♥  = 35  -20 pts
  Q♥  = 36  -30 pts
  K♥  = 37  -40 pts
  A♥  = 38  -50 pts

Grand-slam bonuses  (flat score, replaces all normal scoring for the winner)
─────────────────────────────────────────────────────────────────────────────
  All 13 hearts                              → +200
  All 13 hearts + Q♠                        → +400
  All 13 hearts + Q♠ + J♦                  → +500
  All 13 hearts + Q♠ + J♦ + 10♣           → +1000

Rules (simplified)
──────────────────
  • Any card may be led – no restrictions.
  • Must follow suit if possible; otherwise play any card.
  • Trick won by the highest card of the led suit (no trump).
  • Winner of each trick leads the next.

revealed_hands convention (used by the BeliefPredictor)
────────────────────────────────────────────────────────
  revealed_hands[p, c] = 1   ← observer KNOWS player p holds card c
  revealed_hands[p, c] = 0   ← unknown (not necessarily absent)

  Absence knowledge (e.g., voids, played cards) is encoded
  in the mask (−∞) rather than in revealed_hands.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Card helpers
# ──────────────────────────────────────────────────────────────────────────────

CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3

SUIT_NAMES = {CLUBS: "C", DIAMONDS: "D", HEARTS: "H", SPADES: "S"}
RANK_NAMES = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
              7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}


def make_card(suit: int, rank: int) -> int:
    """Encode a (suit, rank) pair as a single integer.  rank ∈ [2, 14]."""
    return suit * 13 + (rank - 2)


def card_suit(c: int) -> int:
    return c // 13


def card_rank(c: int) -> int:
    """Return integer rank ∈ [2, 14]."""
    return (c % 13) + 2


def card_name(c: int) -> str:
    return RANK_NAMES[c % 13] + SUIT_NAMES[c // 13]


# ──────────────────────────────────────────────────────────────────────────────
# Game constants
# ──────────────────────────────────────────────────────────────────────────────

QUEEN_SPADES  = make_card(SPADES,   12)  # 49
JACK_DIAMONDS = make_card(DIAMONDS, 11)  # 22
TEN_CLUBS     = make_card(CLUBS,    10)  # 8

# suit → frozenset of the 13 card indices in that suit
SUIT_CARDS: Dict[int, frozenset] = {
    s: frozenset(range(s * 13, s * 13 + 13)) for s in range(4)
}

ALL_HEARTS: frozenset = SUIT_CARDS[HEARTS]  # all 13 hearts (2♥ … A♥)

# Per-card base scores used in normal (non-grand-slam) scoring
CARD_BASE_SCORES: Dict[int, int] = {}
for _r in range(5, 11):                               # 5♥–10♥  → -10 each
    CARD_BASE_SCORES[make_card(HEARTS, _r)] = -10
CARD_BASE_SCORES[make_card(HEARTS, 11)] = -20         # J♥
CARD_BASE_SCORES[make_card(HEARTS, 12)] = -30         # Q♥
CARD_BASE_SCORES[make_card(HEARTS, 13)] = -40         # K♥
CARD_BASE_SCORES[make_card(HEARTS, 14)] = -50         # A♥
CARD_BASE_SCORES[QUEEN_SPADES]           = -200        # Q♠
CARD_BASE_SCORES[JACK_DIAMONDS]          = 100         # J♦

FULL_DECK: List[int] = list(range(52))


# ──────────────────────────────────────────────────────────────────────────────
# Score computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_scores(tricks_won: List[Set[int]]) -> List[int]:
    """
    Compute final scores for all 4 players.

    Args:
        tricks_won: tricks_won[p] is the set of card indices won by player p.

    Returns:
        List of 4 integer scores.
    """
    scores = [0, 0, 0, 0]

    for p in range(4):
        hand = tricks_won[p]

        has_all_hearts    = ALL_HEARTS.issubset(hand)
        has_queen_spades  = QUEEN_SPADES  in hand
        has_jack_diamonds = JACK_DIAMONDS in hand
        has_ten_clubs     = TEN_CLUBS     in hand

        # Grand-slam tiers – checked from highest to lowest.
        # The 10♣ contributes to the 1000-point tier only alongside Q♠ + J♦.
        # Outside of a grand-slam, 10♣ acts as a doubler.
        if   has_all_hearts and has_queen_spades and has_jack_diamonds and has_ten_clubs:
            scores[p] = 1000
        elif has_all_hearts and has_queen_spades and has_jack_diamonds:
            scores[p] = 500
        elif has_all_hearts and has_queen_spades:
            scores[p] = 400
        elif has_all_hearts:
            scores[p] = 200
        else:
            base = sum(CARD_BASE_SCORES.get(c, 0) for c in hand)
            if has_ten_clubs:
                scores[p] = 50 if base == 0 else base * 2
            else:
                scores[p] = base

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class GongZhuEnv:
    """
    Stateful 4-player GongZhu environment.

    Typical usage
    ─────────────
        env = GongZhuEnv()
        env.reset()
        while not env.done:
            snap  = env.get_snapshot(observer=0)   # snapshot before the play
            legal = env.legal_actions()
            env.step(random.choice(legal))
        scores = env.score()
    """

    def __init__(self) -> None:
        # Initialise to empty state; call reset() before use.
        self.hands:          List[Set[int]]       = []
        self.tricks_won:     List[Set[int]]        = []
        self.trick_history:  List[Tuple]           = []   # (card, player, trick_num, trick_pos)
        self.current_trick:  List[Tuple[int, int]] = []   # [(card, player), …]
        self.voids:          List[List[bool]]      = []   # voids[player][suit]
        self.current_player: int                   = 0
        self.trick_num:      int                   = 0
        self.done:           bool                  = True  # True until reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, first_player: Optional[int] = None, seed: Optional[int] = None) -> None:
        """
        Shuffle and deal a fresh game.

        Args:
            first_player: Index 0-3 of the player who leads the first trick.
                          If None, chosen uniformly at random.
            seed:         Random seed for deterministic dealing (used by CRN).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        deck = FULL_DECK[:]
        random.shuffle(deck)

        self.hands          = [set(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        self.tricks_won     = [set() for _ in range(4)]
        self.trick_history  = []
        self.current_trick  = []
        self.voids          = [[False] * 4 for _ in range(4)]
        self.trick_num      = 0
        self.done           = False
        self.current_player = (
            first_player if first_player is not None else random.randint(0, 3)
        )

    # ── Core game mechanics ───────────────────────────────────────────────────

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        """
        Return the list of valid card indices the specified player may play.
        Defaults to the current player.
        """
        if player is None:
            player = self.current_player
        hand = self.hands[player]

        if not self.current_trick:
            # Leading: any card is valid
            return list(hand)

        led_suit  = card_suit(self.current_trick[0][0])
        same_suit = [c for c in hand if card_suit(c) == led_suit]
        return same_suit if same_suit else list(hand)

    def step(self, card: int) -> None:
        """
        Play `card` for the current player and advance game state.
        Raises AssertionError if the move is invalid.
        """
        assert not self.done,  "Game is already finished; call reset() first."
        player = self.current_player

        assert card in self.hands[player], (
            f"Card {card_name(card)} is not in player {player}'s hand."
        )
        assert card in self.legal_actions(), (
            f"Card {card_name(card)} is not a legal move for player {player}."
        )

        trick_pos = len(self.current_trick)

        # Detect void: player fails to follow the led suit
        if self.current_trick:
            led_suit    = card_suit(self.current_trick[0][0])
            played_suit = card_suit(card)
            if played_suit != led_suit:
                self.voids[player][led_suit] = True

        # Play the card
        self.hands[player].remove(card)
        self.current_trick.append((card, player))
        self.trick_history.append((card, player, self.trick_num, trick_pos))

        # Resolve trick when all 4 players have played
        if len(self.current_trick) == 4:
            winner = self._trick_winner()
            self.tricks_won[winner].update(c for c, _ in self.current_trick)
            self.current_trick  = []
            self.trick_num     += 1
            self.current_player = winner

            if self.trick_num == 13:
                self.done = True
        else:
            self.current_player = (player + 1) % 4

    def _trick_winner(self) -> int:
        """
        Return the index of the player who wins the current completed trick.
        Winner = highest card of the led suit (no trump suit).
        """
        led_suit  = card_suit(self.current_trick[0][0])
        best_idx  = -1
        winner    = self.current_trick[0][1]
        for c, p in self.current_trick:
            if card_suit(c) == led_suit and c > best_idx:
                best_idx = c
                winner   = p
        return winner

    def score(self) -> List[int]:
        """Return the final 4-player scores. Only valid after the game ends."""
        assert self.done, "Scores are only available after the game ends."
        return compute_scores(self.tricks_won)

    # ── Belief predictor data extraction ──────────────────────────────────────

    def get_revealed_hands(self, observer: int) -> np.ndarray:
        """
        Build the (4, 52) revealed_hands array from `observer`'s perspective.

        The only positive knowledge an observer has is their own hand; voids and
        played-card information belong in the mask.

        Returns:
            np.float32 array of shape (4, 52).
        """
        rh = np.zeros((4, 52), dtype=np.float32)
        for c in self.hands[observer]:
            rh[observer, c] = 1.0
        return rh

    def build_mask(self, observer: int) -> np.ndarray:
        """
        Build the (52, 4) additive logit mask from `observer`'s perspective.

        mask[c, p] = -inf   card c cannot be held by player p
        mask[c, p] = 0.0    card c might be held by player p

        Sources of -inf:
          1. Already-played cards  → entire row -inf
          2. Observer's own cards  → other 3 players cannot hold them
          3. Cards absent from observer's hand (and not played) → observer cannot hold them
          4. Confirmed voids  → player failed to follow suit ⇒ void in that suit

        Returns:
            np.float32 array of shape (52, 4).
        """
        mask    = np.zeros((52, 4), dtype=np.float32)
        NEG_INF = float("-inf")
        played  = {c for c, *_ in self.trick_history}

        # 1 — Played cards
        for c in played:
            mask[c, :] = NEG_INF

        # 2 — Observer's own cards (observer holds them, others don't)
        for c in self.hands[observer]:
            for q in range(4):
                if q != observer:
                    mask[c, q] = NEG_INF

        # 3 — Cards not in observer's hand and not played
        for c in range(52):
            if c not in played and c not in self.hands[observer]:
                mask[c, observer] = NEG_INF

        # 4 — Confirmed voids
        for p in range(4):
            for s in range(4):
                if self.voids[p][s]:
                    for c in SUIT_CARDS[s]:
                        if c not in played:
                            mask[c, p] = NEG_INF

        return mask

    def build_y_true(self) -> np.ndarray:
        """
        Build the (52,) ground-truth label array.

        y_true[c] = player index (0–3) currently holding card c
        y_true[c] = -100 if card c has already been played
                    (PyTorch CrossEntropyLoss ignores index -100 by default)

        Returns:
            np.int64 array of shape (52,).
        """
        y = np.full(52, -100, dtype=np.int64)
        for p in range(4):
            for c in self.hands[p]:
                y[c] = p
        return y

    def get_snapshot(self, observer: int) -> Dict:
        """
        Return a complete training snapshot from `observer`'s perspective.

        Call this BEFORE env.step() to capture the prediction task:
        "given history so far, where is each remaining card?"

        Returns a dict with keys:
            played_cards    np.int64  (T,)        card indices in play order
            players         np.int64  (T,)        who played each card
            trick_nums      np.int64  (T,)        trick index (0–12)
            trick_pos       np.int64  (T,)        position within the trick (0–3)
            revealed_hands  np.float32 (4, 52)
            mask            np.float32 (52, 4)
            y_true          np.int64  (52,)
            seq_len         int                   actual T (for padding masks)
        """
        T = len(self.trick_history)

        if T > 0:
            cards_a   = np.array([h[0] for h in self.trick_history], dtype=np.int64)
            players_a = np.array([h[1] for h in self.trick_history], dtype=np.int64)
            tnums_a   = np.array([h[2] for h in self.trick_history], dtype=np.int64)
            tpos_a    = np.array([h[3] for h in self.trick_history], dtype=np.int64)
        else:
            cards_a = players_a = tnums_a = tpos_a = np.array([], dtype=np.int64)

        return {
            "played_cards":   cards_a,
            "players":        players_a,
            "trick_nums":     tnums_a,
            "trick_pos":      tpos_a,
            "revealed_hands": self.get_revealed_hands(observer),
            "mask":           self.build_mask(observer),
            "y_true":         self.build_y_true(),
            "seq_len":        T,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self) -> "GongZhuEnv":
        """Return a deep copy of the environment (useful for look-ahead / MCTS)."""
        return copy.deepcopy(self)

    def cards_in_play(self) -> Set[int]:
        """Return the set of cards that have already been played."""
        return {c for c, *_ in self.trick_history}

    def __repr__(self) -> str:
        status = (
            "done" if self.done
            else f"trick {self.trick_num}, player {self.current_player} to play"
        )
        return f"<GongZhuEnv  {status}>"
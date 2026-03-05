"""
eval_meta_game.py
─────────────────
Builds the empirical payoff tensor for the current PolicyPool and computes
the AlphaRank meta-Nash distribution over strategies.

The Three Problems This File Addresses
───────────────────────────────────────
1. Combinatorial explosion:
   Naive evaluation of N strategies in a 4-player game requires N^4 matchups.
   GongZhu is symmetric, so only C(N+3,4) unique combos are needed.
   For N=10: 715 matchups vs 10,000.  See build_payoff_tensor().

2. Payoff variance from random deals:
   Trick-taking games have enormous initial-state variance — the deal
   determines most of what can happen.  With 500 games of fresh random deals
   per matchup, a policy that got luckier hands will look stronger than it is,
   producing a noisy payoff tensor and therefore a useless AlphaRank output.

   Fix: Common Random Numbers (CRN).  A shared bank of K random seeds is
   generated once and reused across every matchup in the evaluation.  Every
   matchup plays the same K deals.  Deal-induced variance cancels exactly in
   pairwise comparisons because every policy faces identical card distributions.
   This makes the empirical means much cleaner estimates of true skill deltas
   without needing more games.

   How many games do you actually need with CRN?
   - GongZhu scores range roughly [-200, +400] per player per game.
   - Policy differences at early generations are large (>>20 pts); 200 CRN
     games is often sufficient.
   - By generation 8-10 when policies are close, you may need 1000+.
   - Use --adaptive to let the evaluator decide automatically based on
     per-matchup standard error.

3. Per-matchup standard error tracking:
   Even with CRN, some matchups are high-variance (e.g. all-random vs all-
   random has near-zero expected score difference, huge variance).  The
   evaluator now records per-matchup SE and reports the worst-case SE so you
   know whether to trust the tensor before running AlphaRank.

Usage
─────
  # Standard: 500 CRN games per matchup
  python eval_meta_game.py --save-dir psro_runs --n-games 500

  # Adaptive: stops each matchup when SE < threshold
  python eval_meta_game.py --save-dir psro_runs --adaptive --se-threshold 2.0

  # Fast smoke test
  python eval_meta_game.py --save-dir psro_runs --n-games 100

OpenSpiel
─────────
  pip install open_spiel      # or build from source
  Falls back to uniform distribution if not installed.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from typing import List, Optional

import numpy as np

from base_env import GongZhuEnv
from policy_pool import BaseAgent, PolicyPool

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _progress(iterable, desc: str, disable: bool):
    if not disable and _HAS_TQDM:
        return _tqdm(iterable, desc=desc, dynamic_ncols=True)
    if not disable:
        print(f"[eval] {desc}  ({len(iterable)} matchups)…")
    return iterable


# ──────────────────────────────────────────────────────────────────────────────
# Common Random Numbers seed bank  (Fix 3)
# ──────────────────────────────────────────────────────────────────────────────

def make_crn_seeds(n_games: int, master_seed: int = 42) -> np.ndarray:
    """
    Generate a bank of K integer seeds that are shared across all matchups.

    Every matchup in the evaluation receives this same array of seeds.  Each
    seed deterministically produces one unique card deal + first-player choice.
    Using identical deals across matchups means that any difference in average
    score between two policies is due to their decisions, not their luck.

    This is the standard Common Random Numbers (CRN) variance-reduction
    technique from simulation literature, applied to card games.

    Args:
        n_games:     Number of games (= length of the seed bank).
        master_seed: RNG seed for generating the bank itself.

    Returns:
        seeds: int64 array of shape (n_games,).
    """
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 2**31, size=n_games, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Game simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_matchup(
    agents:    List[BaseAgent],
    crn_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one game per CRN seed and return per-seat mean scores and SEs.

    Each seed deterministically controls both the deal (shuffle) and which
    player leads trick 1.  Using the shared CRN seed bank ensures that all
    matchups in the evaluation are compared on the same set of deals.

    Args:
        agents:    4 BaseAgent callables indexed by seat [0-3].
        crn_seeds: 1-D int64 array of per-game seeds (the shared CRN bank).

    Returns:
        mean_scores: (4,) float64 — mean score per seat across all games.
        se_scores:   (4,) float64 — standard error per seat.
                     SE = std / sqrt(n).  Use seat-0 SE as the matchup's
                     representative noise level when filling the tensor.
    """
    env          = GongZhuEnv()
    n_games      = len(crn_seeds)
    all_scores   = np.zeros((n_games, 4), dtype=np.float64)

    for g, seed in enumerate(crn_seeds):
        rng   = np.random.default_rng(int(seed))
        first = int(rng.integers(0, 4))
        env.reset(first_player=first, seed=int(seed))

        while not env.done:
            p      = env.current_player
            action = agents[p](env)
            env.step(action)

        for p, s in enumerate(env.score()):
            all_scores[g, p] = s

    mean_scores = all_scores.mean(axis=0)
    # SE = sample std / sqrt(n); +1e-8 avoids divide-by-zero in degenerate cases
    se_scores   = all_scores.std(axis=0, ddof=1) / (np.sqrt(n_games) + 1e-8)
    return mean_scores, se_scores


def simulate_matchup_adaptive(
    agents:       List[BaseAgent],
    crn_seeds:    np.ndarray,
    se_threshold: float,
    min_games:    int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Adaptive variant: simulate games in batches until seat-0 SE drops below
    se_threshold or all CRN seeds are exhausted.

    Uses the same CRN seed bank so results remain comparable across matchups.
    Seeds are consumed in order; early-stopping matchups use a prefix of the
    bank, later-stopping ones use more.

    Args:
        agents:       4 BaseAgent callables.
        crn_seeds:    Shared CRN seed bank (upper bound on games).
        se_threshold: Stop when SE of seat-0 falls below this value.
        min_games:    Always simulate at least this many games before checking.

    Returns:
        mean_scores: (4,) float64
        se_scores:   (4,) float64
        n_played:    Actual number of games simulated.
    """
    env         = GongZhuEnv()
    scores_list = []
    batch_size  = 50   # check SE every 50 games

    for batch_start in range(0, len(crn_seeds), batch_size):
        batch = crn_seeds[batch_start: batch_start + batch_size]
        for seed in batch:
            rng   = np.random.default_rng(int(seed))
            first = int(rng.integers(0, 4))
            env.reset(first_player=first, seed=int(seed))

            while not env.done:
                p      = env.current_player
                action = agents[p](env)
                env.step(action)

            scores_list.append([s for s in env.score()])

        n       = len(scores_list)
        arr     = np.array(scores_list, dtype=np.float64)
        se_now  = arr[:, 0].std(ddof=1) / (np.sqrt(n) + 1e-8)

        if n >= min_games and se_now < se_threshold:
            mean_scores = arr.mean(axis=0)
            se_scores   = arr.std(axis=0, ddof=1) / (np.sqrt(n) + 1e-8)
            return mean_scores, se_scores, n

    arr         = np.array(scores_list, dtype=np.float64)
    mean_scores = arr.mean(axis=0)
    se_scores   = arr.std(axis=0, ddof=1) / (np.sqrt(len(scores_list)) + 1e-8)
    return mean_scores, se_scores, len(scores_list)


# ──────────────────────────────────────────────────────────────────────────────
# Payoff tensor construction
# ──────────────────────────────────────────────────────────────────────────────

def build_payoff_tensor(
    pool:         PolicyPool,
    crn_seeds:    np.ndarray,
    verbose:      bool  = True,
    adaptive:     bool  = False,
    se_threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the N×N×N×N empirical payoff tensor with CRN variance reduction.

    M[i, j, k, l] = expected score for a player using strategy i when the
                    four-player lineup is (i, j, k, l).

    Symmetry exploitation
    ─────────────────────
    Only C(N+3,4) unique combos are evaluated (vs N^4 naively).  All valid
    permutations of each evaluated combo are back-filled by the symmetry of
    the expected-score function:  E[score_i | lineup(i,j,k,l)] is the same
    regardless of which physical seat i occupies (averaged over first-player
    randomisation, which CRN already accounts for via the seed bank).

    Duplicate-strategy handling
    ───────────────────────────
    When strategies repeat in a combo (e.g. [A,A,B,C]), multiple permutations
    map to the same tensor cell.  Scores are accumulated and divided by count,
    giving the correct average over all ways that combo can be arranged.

    Variance reporting
    ──────────────────
    The SE tensor (same shape as the payoff tensor) records the standard error
    of the seat-0 score estimate for the unique combo that filled each cell.
    This lets you identify which cells are least trustworthy.

    Args:
        pool:         PolicyPool whose .population is the strategy set.
        crn_seeds:    Shared CRN seed bank from make_crn_seeds().
        verbose:      Print progress and summary stats.
        adaptive:     Use adaptive stopping based on per-matchup SE.
        se_threshold: SE target for adaptive mode.

    Returns:
        payoff_tensor: (N,N,N,N) float64 — mean score.
        se_tensor:     (N,N,N,N) float64 — seat-0 SE for the filling matchup.
    """
    N      = len(pool)
    agents = pool.population
    combos = list(itertools.combinations_with_replacement(range(N), 4))
    n_games_planned = len(crn_seeds)

    if verbose:
        n_naive = N ** 4
        print(
            f"\n[eval] Payoff tensor  N={N}  "
            f"unique matchups={len(combos):,} (naive={n_naive:,}  "
            f"reduction={n_naive / len(combos):.0f}×)"
        )
        if adaptive:
            print(
                f"[eval] Mode: adaptive  SE threshold={se_threshold:.2f}  "
                f"max games/matchup={n_games_planned}"
            )
        else:
            print(
                f"[eval] Mode: fixed  games/matchup={n_games_planned}  "
                f"total≈{len(combos) * n_games_planned:,}"
            )
        print(f"[eval] Variance reduction: Common Random Numbers (shared {n_games_planned}-game seed bank)")

    M_sum  = np.zeros([N] * 4, dtype=np.float64)
    M_cnt  = np.zeros([N] * 4, dtype=np.int64)
    SE_max = np.zeros([N] * 4, dtype=np.float64)  # worst-case SE per cell

    total_games_played = 0

    for combo in _progress(combos, desc="Evaluating matchups", disable=not verbose):
        matchup_agents = [agents[idx] for idx in combo]

        if adaptive:
            mean_scores, se_scores, n_played = simulate_matchup_adaptive(
                matchup_agents, crn_seeds,
                se_threshold=se_threshold,
            )
            total_games_played += n_played
        else:
            mean_scores, se_scores = simulate_matchup(matchup_agents, crn_seeds)
            total_games_played += len(crn_seeds)

        # Backfill all permutations of this combo's strategy indices.
        # For permutation perm, seat perm[s] plays strategy combo[s].
        # M[combo[perm[0]], combo[perm[1]], ...] = score earned by seat perm[0].
        for perm in itertools.permutations(range(4)):
            key          = tuple(combo[p] for p in perm)
            M_sum[key]  += mean_scores[perm[0]]
            M_cnt[key]  += 1
            # Track the largest SE seen for each cell (conservative)
            SE_max[key]  = max(SE_max[key], se_scores[0])

    # Average accumulated values; cells with count=0 stay 0 (never happens)
    M = np.divide(M_sum, M_cnt, where=M_cnt > 0, out=np.zeros_like(M_sum))

    if verbose:
        worst_se   = SE_max[M_cnt > 0].max()
        median_se  = np.median(SE_max[M_cnt > 0])
        print(
            f"[eval] Tensor complete.  "
            f"Score range: [{M.min():.1f}, {M.max():.1f}]  "
            f"Mean: {M.mean():.1f}"
        )
        print(
            f"[eval] SE (seat-0):  worst={worst_se:.2f}  median={median_se:.2f}  "
            f"total games played: {total_games_played:,}"
        )
        if worst_se > 5.0:
            print(
                f"[eval] WARNING: worst-case SE={worst_se:.2f} is high.  "
                f"Consider increasing --n-games (or using --adaptive) for a "
                f"more reliable payoff tensor."
            )

    return M, SE_max


# ──────────────────────────────────────────────────────────────────────────────
# AlphaRank solver
# ──────────────────────────────────────────────────────────────────────────────

def run_alpharank(
    payoff_tensor: np.ndarray,
    se_tensor:     np.ndarray | None = None,
    alpha:         float = 1e-2,
    verbose:       bool  = True,
) -> np.ndarray:
    """
    Run OpenSpiel's AlphaRank on the (N,N,N,N) payoff tensor.

    Returns the marginal meta-Nash distribution as a (N,) probability vector.

    AlphaRank models an evolutionary Markov chain where agents switch to
    better-performing strategies with rates proportional to payoff advantage.
    The stationary distribution of this chain is the meta-Nash.

    For a 4-player symmetric game, alpharank() returns pi over N^4 strategy
    profiles.  We marginalise: μ[i] = Σ_{j,k,l} π[i,j,k,l].

    Falls back to uniform if OpenSpiel is unavailable or the solver fails.

    Args:
        payoff_tensor: (N,N,N,N) from build_payoff_tensor().
        se_tensor:     (N,N,N,N) SE estimates.  If provided and worst SE is
                       high, a warning is printed before running the solver.
        alpha:         Selection pressure temperature.
        verbose:       Print convergence info and distribution.

    Returns:
        distribution: (N,) float64 summing to 1.
    """
    N = payoff_tensor.shape[0]

    if N == 1:
        return np.array([1.0])

    if se_tensor is not None and verbose:
        worst_se = se_tensor.max()
        if worst_se > 5.0:
            print(
                f"[eval] AlphaRank input has worst SE={worst_se:.2f}.  "
                f"The distribution may be unreliable — consider re-evaluating "
                f"with more games."
            )

    try:
        from open_spiel.python.egt import alpharank as ar
    except ImportError:
        if verbose:
            print(
                "[eval] OpenSpiel not found — returning uniform distribution.\n"
                "       Install with: pip install open_spiel"
            )
        return np.ones(N) / N

    try:
        payoff_tables = [payoff_tensor] * 4
        
        # FIX: The correct function is compute(), which returns 5 values.
        rhos, rho_m, pi, num_profiles, num_strats_per_pop = ar.compute(payoff_tables, alpha=alpha)

        # Marginalise: pi is a flat (N^4,) vector over all strategy profiles
        pi_4d    = np.array(pi).reshape([N] * 4)
        marginal = pi_4d.reshape(N, -1).sum(axis=1)
        marginal = np.clip(marginal, 0.0, 1.0) # clip to 0 before normalization
        marginal /= marginal.sum()   # renormalise for floating-point safety

        if verbose:
            print(f"[eval] AlphaRank converged. (alpha={alpha})")

        return marginal

    except Exception as exc:
        if verbose:
            print(f"[eval] AlphaRank solver error ({exc!r}) — returning uniform.")
        return np.ones(N) / N


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    pool:         PolicyPool,
    save_dir:     str,
    n_games:      int   = 500,
    alpha:        float = 1e-2,
    master_seed:  int   = 42,
    adaptive:     bool  = False,
    se_threshold: float = 2.0,
    verbose:      bool  = True,
) -> np.ndarray:
    """
    Full evaluation pipeline:
        CRN seed bank → simulate → payoff tensor → AlphaRank → save → return.

    Output files written to save_dir:
        payoff_tensor.npy            — raw (N,N,N,N) payoff tensor
        se_tensor.npy                — per-cell standard error tensor
        alpharank_distribution.json  — distribution + metadata

    Args:
        pool:         PolicyPool to evaluate.
        save_dir:     Directory for output files.
        n_games:      Games per matchup (fixed mode) or max games (adaptive).
        alpha:        AlphaRank temperature.
        master_seed:  Seed for generating the CRN bank.
        adaptive:     Use adaptive stopping per matchup based on SE.
        se_threshold: SE target for adaptive mode.
        verbose:      Print progress and summary.

    Returns:
        distribution: (N,) AlphaRank meta-Nash weights.
    """
    os.makedirs(save_dir, exist_ok=True)
    N  = len(pool)
    t0 = time.time()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Meta-game Evaluation   pool size={N}")
        print(f"{'─'*60}")

    # ── Step 1: CRN seed bank ─────────────────────────────────────────────────
    crn_seeds = make_crn_seeds(n_games, master_seed=master_seed)
    if verbose:
        print(f"[eval] CRN bank: {n_games} seeds (master_seed={master_seed})")

    # ── Step 2: Build payoff tensor ───────────────────────────────────────────
    payoff_tensor, se_tensor = build_payoff_tensor(
        pool,
        crn_seeds    = crn_seeds,
        verbose      = verbose,
        adaptive     = adaptive,
        se_threshold = se_threshold,
    )

    np.save(os.path.join(save_dir, "payoff_tensor.npy"), payoff_tensor)
    np.save(os.path.join(save_dir, "se_tensor.npy"),     se_tensor)
    if verbose:
        print(f"[eval] Payoff tensor  → {save_dir}/payoff_tensor.npy")
        print(f"[eval] SE tensor      → {save_dir}/se_tensor.npy")

    # ── Step 3: AlphaRank ─────────────────────────────────────────────────────
    distribution = run_alpharank(
        payoff_tensor, se_tensor=se_tensor, alpha=alpha, verbose=verbose
    )

    # ── Step 4: Save distribution ─────────────────────────────────────────────
    elapsed   = time.time() - t0
    dist_path = os.path.join(save_dir, "alpharank_distribution.json")
    with open(dist_path, "w") as f:
        json.dump(
            {
                "distribution":  distribution.tolist(),
                "n_agents":      N,
                "n_games":       n_games,
                "adaptive":      adaptive,
                "se_threshold":  se_threshold if adaptive else None,
                "worst_se":      float(se_tensor.max()),
                "median_se":     float(np.median(se_tensor[se_tensor > 0])),
                "alpha":         alpha,
                "master_seed":   master_seed,
                "elapsed_s":     round(elapsed, 1),
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f, indent=2,
        )
    if verbose:
        print(f"[eval] Distribution   → {dist_path}")

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    if verbose:
        print(f"\n  {'Agent':<16} {'Weight':>7}  Histogram")
        print(f"  {'─'*16} {'─'*7}  {'─'*36}")
        for i, (agent, w) in enumerate(zip(pool.population, distribution)):
            bar   = "█" * max(1, int(w * 36))
            label = f"[{i}] {type(agent).__name__}"
            print(f"  {label:<18} {w:>6.3f}  {bar}")
        print(f"\n  Worst SE={se_tensor.max():.2f}  "
              f"Median SE={np.median(se_tensor[se_tensor > 0]):.2f}  "
              f"Elapsed={elapsed/60:.1f} min")
        print(f"{'─'*60}\n")

    return distribution


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from run_psro import _load_pool, _save_pool_state, _current_gen

    p = argparse.ArgumentParser(
        description="Evaluate PSRO meta-game with AlphaRank.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Standard: 500 CRN games per matchup
  python eval_meta_game.py --save-dir psro_runs --n-games 500

  # Adaptive: stop each matchup when SE < 2.0 (up to 2000 games)
  python eval_meta_game.py --save-dir psro_runs --adaptive --n-games 2000 --se-threshold 2.0

  # Fast smoke test
  python eval_meta_game.py --save-dir psro_runs --n-games 100
        """,
    )
    p.add_argument("--save-dir",     type=str,   default="psro_runs")
    p.add_argument("--n-games",      type=int,   default=500,
                   help="Games per matchup (fixed) or max games (adaptive).")
    p.add_argument("--alpha",        type=float, default=1e-2,
                   help="AlphaRank temperature.")
    p.add_argument("--master-seed",  type=int,   default=42,
                   help="Master seed for the CRN bank.")
    p.add_argument("--adaptive",     action="store_true",
                   help="Adaptive stopping: stop each matchup when SE < se-threshold.")
    p.add_argument("--se-threshold", type=float, default=2.0,
                   help="SE target for adaptive mode (score units).")
    p.add_argument("--no-save-pool", action="store_true",
                   help="Print distribution but do not update pool_state.json.")
    args = p.parse_args()

    pool, state = _load_pool(args.save_dir, device="cpu")
    gen         = _current_gen(state)
    eval_dir    = os.path.join(args.save_dir, f"eval_gen_{gen:03d}")

    dist = evaluate(
        pool         = pool,
        save_dir     = eval_dir,
        n_games      = args.n_games,
        alpha        = args.alpha,
        master_seed  = args.master_seed,
        adaptive     = args.adaptive,
        se_threshold = args.se_threshold,
    )

    if not args.no_save_pool:
        pool.set_distribution(dist)
        state["alpharank_distribution"] = dist.tolist()
        _save_pool_state(pool, state, args.save_dir)
        print("[eval] Distribution written to pool_state.json.")
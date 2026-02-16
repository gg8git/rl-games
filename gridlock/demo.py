"""
Interactive Policy Demo - Visualize trained policy decisions

This script lets you:
1. Watch the policy play complete games step-by-step
2. See the policy's action probabilities and value estimates
3. Understand why the policy makes each decision
4. Compare different policies (SFT vs PPO)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import time

# Import from training script
from ppo_sft_final import (
    PPOPolicy, State, Square, validate_action, no_valid_moves, 
    score, sample_draw_batch, device
)

State = namedtuple('State', ('grid', 'num'))


def print_grid_fancy(grid: np.ndarray, highlight_pos=None):
    """
    Pretty print the grid with optional highlighting
    
    Args:
        grid: 3x3 numpy array
        highlight_pos: (row, col) to highlight in green
    """
    # Color codes
    RESET = "\033[0m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    
    print("\n     0     1     2")
    print("  ┌─────┬─────┬─────┐")
    
    for i in range(3):
        row_str = f"{BLUE}{i}{RESET} │"
        for j in range(3):
            val = grid[i, j]
            
            # Highlight if this is the new position
            if highlight_pos and highlight_pos == (i, j):
                if val == 0:
                    cell = f"{GREEN}  -  {RESET}"
                else:
                    cell = f"{GREEN} {val:2d}  {RESET}"
            else:
                if val == 0:
                    cell = "  -  "
                else:
                    cell = f" {val:2d}  "
            
            row_str += cell + "│"
        
        print(row_str)
        
        if i < 2:
            print("  ├─────┼─────┼─────┤")
        else:
            print("  └─────┴─────┴─────┘")
    
    print()


def get_action_probabilities(policy: PPOPolicy, state: State):
    """
    Get action probabilities and value estimate for a state
    
    Returns:
        probs: numpy array of shape (9,) with action probabilities
        value: predicted state value
        valid_mask: which actions are valid
    """
    state_encoding = policy.encode_state(state.grid, state.num)
    logits, value = policy.get_action_and_value(state_encoding.unsqueeze(0))
    
    # Get valid actions
    valid_mask = policy.get_valid_action_mask(state.grid, state.num)
    valid_mask_tensor = torch.FloatTensor(valid_mask).to(device)
    
    # Apply masking
    masked_logits = logits.clone()
    masked_logits[0, valid_mask_tensor == 0] = float('-inf')
    
    # Convert to probabilities
    probs = F.softmax(masked_logits, dim=1)[0].cpu().detach().numpy()
    
    return probs, value.item(), valid_mask


def print_action_distribution(probs: np.ndarray, valid_mask: np.ndarray, value: float):
    """
    Print action probabilities in a grid layout
    
    Args:
        probs: action probabilities (9,)
        valid_mask: valid action mask (9,)
        value: value estimate
    """
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    
    print(f"\n{BOLD}Policy Analysis:{RESET}")
    print(f"  Value Estimate: {value:.2f}")
    print(f"\n  Action Probabilities (grid layout):")
    print("     0        1        2")
    print("  ┌────────┬────────┬────────┐")
    
    for i in range(3):
        row_str = f"  │"
        for j in range(3):
            idx = i * 3 + j
            prob = probs[idx]
            is_valid = valid_mask[idx] > 0
            
            if is_valid:
                if prob > 0.3:  # High probability
                    row_str += f"{GREEN}{prob:6.1%}{RESET}  │"
                elif prob > 0.1:  # Medium probability
                    row_str += f"{prob:6.1%}  │"
                else:  # Low probability
                    row_str += f"{GRAY}{prob:6.1%}{RESET}  │"
            else:
                row_str += f"{RED}    X{RESET}    │"
        
        print(row_str)
        
        if i < 2:
            print("  ├────────┼────────┼────────┤")
        else:
            print("  └────────┴────────┴────────┘")
    
    # Show top 3 choices
    valid_probs = [(i, probs[i]) for i in range(9) if valid_mask[i] > 0]
    valid_probs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  {BOLD}Top Choices:{RESET}")
    for rank, (idx, prob) in enumerate(valid_probs[:3], 1):
        row, col = idx // 3, idx % 3
        print(f"    {rank}. Position ({row}, {col}): {prob:.1%}")
    
    print()


def play_game_interactive(
    policy: PPOPolicy, 
    draw: np.ndarray = None,
    step_by_step: bool = True,
    show_probabilities: bool = True
):
    """
    Play a game with the policy and show decision-making process
    
    Args:
        policy: Trained policy
        draw: Card sequence (if None, random)
        step_by_step: If True, wait for user input between steps
        show_probabilities: If True, show action probabilities
    """
    if draw is None:
        draw = sample_draw_batch(1)[0]
    
    grid = np.zeros((3, 3), dtype=np.int32)
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    
    print(f"\n{'='*80}")
    print(f"{BOLD}{CYAN}POLICY PLAYTHROUGH DEMO{RESET}")
    print(f"{'='*80}")
    print(f"\nCard sequence (first 15): {draw[:15]}")
    print(f"Full deck: {draw}")
    
    total_score = 0
    
    for turn in range(min(9, len(draw))):
        current_card = int(draw[turn])
        if no_valid_moves(grid, turn):
            print(f"\n{BOLD}❌ No more valid moves!{RESET}")
            print(f"\n{BOLD}Game Over{RESET}")
            break

        state = State(grid.copy(), current_card)
        
        print(f"\n{BOLD}{'─'*80}{RESET}")
        print(f"{BOLD}{YELLOW}Turn {turn + 1}/9{RESET}")
        print(f"{BOLD}{'─'*80}{RESET}")
        print(f"\n{BOLD}Current Card: {current_card}{RESET}")
        
        # Show current board
        print(f"\n{BOLD}Current Board:{RESET}")
        print_grid_fancy(grid)
        
        # Get policy's decision
        probs, value, valid_mask = get_action_probabilities(policy, state)
        
        # Show analysis if requested
        if show_probabilities:
            print_action_distribution(probs, valid_mask, value)
        
        # Sample action
        action = policy.sample_action(state, exploration_rate=0.0, argmax=True)
        square = Square(action.idx)
        
        # Check if valid
        is_valid = validate_action(grid, current_card, square)
        
        if not is_valid:
            print(f"\n{BOLD}❌ Invalid move attempted!{RESET}")
            print(f"   Tried to place {current_card} at position ({square.row}, {square.col})")
            print(f"   This should not happen with a well-trained policy!")
            print(f"\n{BOLD}Game Over - No valid moves{RESET}")
            break
        
        # Make the move
        grid[square.row, square.col] = current_card
        action_prob = probs[action.idx]
        
        print(f"\n{BOLD}✓ Policy Decision:{RESET}")
        print(f"   Placed {current_card} at position ({square.row}, {square.col})")
        print(f"   Confidence: {action_prob:.1%}")
        
        # Show updated board
        print(f"\n{BOLD}Board After Move:{RESET}")
        print_grid_fancy(grid, highlight_pos=(square.row, square.col))
        
        # Show current score
        current_score = score(grid)
        print(f"{BOLD}Current Score: {current_score}{RESET}")
        
        # Wait for user if step-by-step
        if step_by_step and turn < min(8, len(draw) - 1):
            input(f"\n{CYAN}Press Enter to continue...{RESET}")
    
    # Final summary
    final_score = score(grid)
    
    print(f"\n{'='*80}")
    print(f"{BOLD}GAME COMPLETE{RESET}")
    print(f"{'='*80}")
    print(f"\n{BOLD}Final Board:{RESET}")
    print_grid_fancy(grid)
    
    # Show scoring breakdown
    print(f"{BOLD}Scoring Breakdown:{RESET}")
    
    # Count complete structures
    complete_rows = sum((grid != 0).all(axis=1))
    complete_cols = sum((grid != 0).all(axis=0))
    
    main_diag = np.diag(grid)
    main_diag_complete = (main_diag != 0).all()
    
    anti_diag = np.diag(np.fliplr(grid))
    anti_diag_complete = (anti_diag != 0).all()
    
    print(f"  Complete rows: {complete_rows}")
    print(f"  Complete cols: {complete_cols}")
    print(f"  Main diagonal: {'✓' if main_diag_complete else '✗'}")
    print(f"  Anti diagonal: {'✓' if anti_diag_complete else '✗'}")
    
    print(f"\n{BOLD}Final Score: {final_score}{RESET}")
    print(f"{'='*80}\n")
    
    return final_score


def compare_policies(
    policy1: PPOPolicy,
    policy2: PPOPolicy,
    policy1_name: str = "Policy 1",
    policy2_name: str = "Policy 2",
    num_games: int = 100
):
    """
    Compare two policies by having them play the same card sequences
    
    Args:
        policy1: First policy
        policy2: Second policy
        policy1_name: Name for first policy
        policy2_name: Name for second policy
        num_games: Number of games to compare
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    
    print(f"\n{'='*80}")
    print(f"{BOLD}POLICY COMPARISON{RESET}")
    print(f"{'='*80}")
    print(f"Comparing {policy1_name} vs {policy2_name} on {num_games} games\n")
    
    draws = sample_draw_batch(num_games)
    
    policy1_scores = []
    policy2_scores = []
    policy1_wins = 0
    policy2_wins = 0
    ties = 0
    
    for i in range(num_games):
        # Policy 1
        grid1 = np.zeros((3, 3), dtype=np.int32)
        for turn in range(min(9, len(draws[i]))):
            state = State(grid1.copy(), int(draws[i][turn]))
            action = policy1.sample_action(state, exploration_rate=0.0)
            square = Square(action.idx)
            if not validate_action(grid1, draws[i][turn], square):
                break
            grid1[square.row, square.col] = draws[i][turn]
        score1 = score(grid1)
        policy1_scores.append(score1)
        
        # Policy 2
        grid2 = np.zeros((3, 3), dtype=np.int32)
        for turn in range(min(9, len(draws[i]))):
            state = State(grid2.copy(), int(draws[i][turn]))
            action = policy2.sample_action(state, exploration_rate=0.0)
            square = Square(action.idx)
            if not validate_action(grid2, draws[i][turn], square):
                break
            grid2[square.row, square.col] = draws[i][turn]
        score2 = score(grid2)
        policy2_scores.append(score2)
        
        # Count wins
        if score1 > score2:
            policy1_wins += 1
        elif score2 > score1:
            policy2_wins += 1
        else:
            ties += 1
    
    # Print results
    avg1 = np.mean(policy1_scores)
    avg2 = np.mean(policy2_scores)
    
    print(f"{BOLD}Results:{RESET}\n")
    print(f"  {BLUE}{policy1_name}:{RESET}")
    print(f"    Average score: {avg1:.2f}")
    print(f"    Max score: {np.max(policy1_scores)}")
    print(f"    Min score: {np.min(policy1_scores)}")
    print(f"    Wins: {policy1_wins} ({100*policy1_wins/num_games:.1f}%)")
    
    print(f"\n  {GREEN}{policy2_name}:{RESET}")
    print(f"    Average score: {avg2:.2f}")
    print(f"    Max score: {np.max(policy2_scores)}")
    print(f"    Min score: {np.min(policy2_scores)}")
    print(f"    Wins: {policy2_wins} ({100*policy2_wins/num_games:.1f}%)")
    
    print(f"\n  Ties: {ties} ({100*ties/num_games:.1f}%)")
    
    print(f"\n{BOLD}Winner: ", end="")
    if avg1 > avg2:
        print(f"{BLUE}{policy1_name}{RESET} by {avg1-avg2:.2f} points")
    elif avg2 > avg1:
        print(f"{GREEN}{policy2_name}{RESET} by {avg2-avg1:.2f} points")
    else:
        print(f"Tie!{RESET}")
    
    print(f"{'='*80}\n")


def play_multiple_games_summary(policy: PPOPolicy, num_games: int = 10):
    """
    Play multiple games and show summary statistics
    
    Args:
        policy: Policy to evaluate
        num_games: Number of games to play
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    print(f"\n{'='*80}")
    print(f"{BOLD}PLAYING {num_games} GAMES{RESET}")
    print(f"{'='*80}\n")
    
    draws = sample_draw_batch(num_games)
    scores = []
    lengths = []
    
    for i in range(num_games):
        grid = np.zeros((3, 3), dtype=np.int32)
        num_moves = 0
        
        for turn in range(min(9, len(draws[i]))):
            state = State(grid.copy(), int(draws[i][turn]))
            action = policy.sample_action(state, exploration_rate=0.0)
            square = Square(action.idx)
            if not validate_action(grid, draws[i][turn], square):
                break
            grid[square.row, square.col] = draws[i][turn]
            num_moves += 1
        
        final_score = score(grid)
        scores.append(final_score)
        lengths.append(num_moves)
        
        print(f"Game {i+1:2d}: Score = {final_score:3d}, Cards placed = {num_moves}")
    
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Std dev: {np.std(scores):.2f}")
    print(f"  Max score: {np.max(scores)}")
    print(f"  Min score: {np.min(scores)}")
    print(f"  Average cards placed: {np.mean(lengths):.1f}")
    print(f"{'='*80}\n")


def interactive_menu():
    """Main interactive menu"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    
    print(f"\n{BOLD}{CYAN}{'='*80}")
    print(f"POLICY DEMO - INTERACTIVE MENU")
    print(f"{'='*80}{RESET}\n")
    
    # Load policies
    policies = {}
    policy_names = []
    
    print("Loading policies...\n")
    
    # Try to load different policies
    policy_files = [
        ('./ckpt/ppo_sft_final/sft_actor_only.pt', 'SFT Actor'),
        ('./ckpt/ppo_sft_final/sft_with_critic.pt', 'SFT + Critic'),
        ('./ckpt/ppo_sft_final/best_ppo_policy.pt', 'PPO Best'),
        ('./ckpt/ppo_sft_final/final_policy.pt', 'PPO Final'),
    ]
    
    for path, name in policy_files:
        try:
            policy = PPOPolicy()
            policy.load(path)
            policies[name] = policy
            policy_names.append(name)
            print(f"  ✓ Loaded: {name}")
        except FileNotFoundError:
            print(f"  ✗ Not found: {name} ({path})")
    
    if len(policies) == 0:
        print("\n❌ No policies found! Please train a policy first.")
        return
    
    print(f"\n{len(policies)} policies loaded.\n")
    
    # Select policy
    print(f"{BOLD}Available policies:{RESET}")
    for i, name in enumerate(policy_names, 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect policy (1-{len(policy_names)}): "))
            if 1 <= choice <= len(policy_names):
                selected_policy_name = policy_names[choice - 1]
                selected_policy = policies[selected_policy_name]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    print(f"\n{BOLD}Selected: {selected_policy_name}{RESET}\n")
    
    # Main menu loop
    while True:
        print(f"\n{BOLD}{CYAN}{'='*80}")
        print(f"MAIN MENU")
        print(f"{'='*80}{RESET}")
        print(f"1. Watch policy play ONE game (step-by-step, detailed)")
        print(f"2. Watch policy play ONE game (auto-play, detailed)")
        print(f"3. Play 10 games (summary only)")
        print(f"4. Play 100 games (statistics)")
        print(f"5. Compare with another policy")
        print(f"6. Switch policy")
        print(f"7. Exit")
        
        try:
            choice = input(f"\n{BOLD}Choose option (1-7): {RESET}")
            
            if choice == '1':
                play_game_interactive(selected_policy, step_by_step=True, show_probabilities=True)
            
            elif choice == '2':
                play_game_interactive(selected_policy, step_by_step=False, show_probabilities=True)
            
            elif choice == '3':
                play_multiple_games_summary(selected_policy, num_games=10)
            
            elif choice == '4':
                play_multiple_games_summary(selected_policy, num_games=100)
            
            elif choice == '5':
                if len(policies) < 2:
                    print("\n❌ Need at least 2 policies to compare!")
                    continue
                
                print(f"\n{BOLD}Compare with:{RESET}")
                other_names = [name for name in policy_names if name != selected_policy_name]
                for i, name in enumerate(other_names, 1):
                    print(f"  {i}. {name}")
                
                try:
                    comp_choice = int(input(f"\nSelect policy (1-{len(other_names)}): "))
                    if 1 <= comp_choice <= len(other_names):
                        compare_name = other_names[comp_choice - 1]
                        compare_policy = policies[compare_name]
                        compare_policies(
                            selected_policy, 
                            compare_policy,
                            selected_policy_name,
                            compare_name,
                            num_games=100
                        )
                except (ValueError, IndexError):
                    print("Invalid choice!")
            
            elif choice == '6':
                print(f"\n{BOLD}Available policies:{RESET}")
                for i, name in enumerate(policy_names, 1):
                    print(f"  {i}. {name}")
                
                try:
                    new_choice = int(input(f"\nSelect policy (1-{len(policy_names)}): "))
                    if 1 <= new_choice <= len(policy_names):
                        selected_policy_name = policy_names[new_choice - 1]
                        selected_policy = policies[selected_policy_name]
                        print(f"\n{BOLD}Switched to: {selected_policy_name}{RESET}")
                except ValueError:
                    print("Invalid choice!")
            
            elif choice == '7':
                print(f"\n{BOLD}Goodbye!{RESET}\n")
                break
            
            else:
                print("\n❌ Invalid option! Please choose 1-7.")
        
        except KeyboardInterrupt:
            print(f"\n\n{BOLD}Goodbye!{RESET}\n")
            break


if __name__ == "__main__":
    interactive_menu()
"""
Interactive Demo for CleanRL PPO Agent

This script allows you to:
1. Watch a trained CleanRL agent play Gridlock step-by-step
2. Inspect action probabilities and value estimates
3. Compare two trained agents
4. Run summary statistics over many games

Works with:
    best_agent.pt
    final_agent.pt
"""

import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from gridlock.cleanrl_ppo_sft import Agent, make_env
import gymnasium as gym


# ============================================================
# Utility: Pretty Grid Printer
# ============================================================

def print_grid(grid, highlight=None):
    RESET = "\033[0m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"

    print("\n     0     1     2")
    print("  ┌─────┬─────┬─────┐")

    for i in range(3):
        row_str = f"{BLUE}{i}{RESET} │"
        for j in range(3):
            val = grid[i, j]
            if highlight == (i, j):
                row_str += f"{GREEN}{val:3d}{RESET}  │"
            else:
                row_str += f"{val:3d}  │"
        print(row_str)
        if i < 2:
            print("  ├─────┼─────┼─────┤")
        else:
            print("  └─────┴─────┴─────┘")


# ============================================================
# Action Analysis
# ============================================================

def get_action_info(agent, obs, mask, device, argmax=True):
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        mask_t = torch.tensor(mask).unsqueeze(0).to(device)

        logits = agent.actor(obs_t)
        masked_logits = torch.where(mask_t, logits, torch.tensor(float("-inf")).to(device))
        probs = F.softmax(masked_logits, dim=-1)

        if argmax:
            action = masked_logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        value = agent.critic(obs_t)

    return action.item(), probs.squeeze(0).cpu().numpy(), value.item()


def print_action_distribution(probs, mask, value):
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}Policy Analysis{RESET}")
    print(f"Value estimate: {value:.2f}\n")

    print("     0        1        2")
    print("  ┌────────┬────────┬────────┐")

    for i in range(3):
        row_str = "  │"
        for j in range(3):
            idx = i * 3 + j
            if not mask[idx]:
                row_str += f"{RED}   X   {RESET}│"
            else:
                p = probs[idx]
                if p > 0.3:
                    row_str += f"{GREEN}{p:6.1%}{RESET}│"
                elif p > 0.1:
                    row_str += f"{p:6.1%}│"
                else:
                    row_str += f"{GRAY}{p:6.1%}{RESET}│"
        print(row_str)

        if i < 2:
            print("  ├────────┼────────┼────────┤")
        else:
            print("  └────────┴────────┴────────┘")


# ============================================================
# Single Game Playthrough
# ============================================================

def play_game(agent, device, step_by_step=True, argmax=True):
    env = make_env()()
    obs, info = env.reset()

    done = False

    print("\n" + "=" * 80)
    print("CLEANRL POLICY PLAYTHROUGH")
    print("=" * 80)

    while not done:
        mask = env.unwrapped.get_action_mask()
        action, probs, value = get_action_info(agent, obs, mask, device, argmax)

        grid = env.unwrapped.grid.copy()
        current_card = int(obs[-1] * 10)

        print(f"\nCurrent Card: {current_card}")
        print_grid(grid)
        print_action_distribution(probs, mask, value)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        grid_after = env.unwrapped.grid.copy()
        row, col = action // 3, action % 3

        print(f"\nPlaced at ({row}, {col})")
        print_grid(grid_after, highlight=(row, col))

        if step_by_step and not done:
            input("Press Enter to continue...")

        if done:
            print("\nGame Over")

            base_env = env.unwrapped

            if base_env.pointer >= 9:
                print("Cause: Grid completed (9 cards placed)")
            elif base_env.pointer < 40:
                # This is the deadlock card
                deadlock_card = base_env.deck[base_env.pointer]
                print(f"Cause: No valid moves for next card: {deadlock_card}")
                print(f"Whole deck: {base_env.deck}")
                print(f"Rest of deck: {base_env.deck[base_env.pointer:]}")
            else:
                print("Cause: Unknown termination condition")

            print(f"Final Score: {reward}")

    env.close()


# ============================================================
# Multi-Game Summary
# ============================================================

def evaluate(agent, device, n_games=50, argmax=True):
    scores = []

    for _ in range(n_games):
        env = make_env()()
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.unwrapped.get_action_mask()
            action, _, _ = get_action_info(agent, obs, mask, device, argmax)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        scores.append(reward)
        env.close()

    print("\nSummary")
    print(f"Games: {n_games}")
    print(f"Mean Score: {np.mean(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")


# ============================================================
# Policy Comparison
# ============================================================

def compare(agent1, agent2, device, n_games=100):
    scores1, scores2 = [], []

    for _ in range(n_games):
        env = make_env()()
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.unwrapped.get_action_mask()
            a1, _, _ = get_action_info(agent1, obs, mask, device, True)
            a2, _, _ = get_action_info(agent2, obs, mask, device, True)

            # Step env twice separately for fairness
            obs1, r1, t1, tr1, _ = env.step(a1)
            obs2, r2, t2, tr2, _ = env.step(a2)

            done = t1 or tr1

        scores1.append(r1)
        scores2.append(r2)
        env.close()

    print("\nComparison Results")
    print(f"Agent 1 Mean: {np.mean(scores1):.2f}")
    print(f"Agent 2 Mean: {np.mean(scores2):.2f}")


# ============================================================
# Load Agent
# ============================================================

def load_agent(path, device):
    dummy_env = gym.vector.SyncVectorEnv([make_env()])
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(torch.load(path, map_location=device))
    agent.eval()
    return agent


# ============================================================
# Interactive Menu
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nAvailable model files:")
    models = []
    for root, _, files in os.walk("runs"):
        for f in files:
            if f.endswith(".pt"):
                models.append(os.path.join(root, f))

    for i, m in enumerate(models):
        print(f"{i+1}. {m}")

    choice = int(input("\nSelect model: ")) - 1
    agent = load_agent(models[choice], device)

    while True:
        print("\nOptions:")
        print("1. Step-by-step game")
        print("2. Auto-play game")
        print("3. Evaluate 50 games")
        print("4. Exit")

        c = input("Choice: ")

        if c == "1":
            play_game(agent, device, step_by_step=True, argmax=True)
        elif c == "2":
            play_game(agent, device, step_by_step=False, argmax=True)
        elif c == "3":
            evaluate(agent, device, n_games=50)
        elif c == "4":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()

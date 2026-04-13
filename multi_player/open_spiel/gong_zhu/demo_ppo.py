"""
demo_ppo.py
───────────
Play against your trained PSRO models in the terminal.

Features:
- "Simulation Mode" (Auto-dealt hands, you play against bots)
- "Live Play Mode" (You play with physical cards, inputting the game state)
- Support for specific generations: use "ppo_1", "ppo_5", etc.
- Uses the _ModelCache to prevent RAM bloat even if playing against 3 different generations.

Usage:
  python demo_ppo.py --save-dir psro_runs --players human,ppo,ppo_1,random
"""

import argparse
import time
import torch
import numpy as np

from base_env import GongZhuEnv, card_name, card_suit, CARD_BASE_SCORES, TEN_CLUBS
from run_psro import _load_pool

def parse_card_string(card_str: str) -> int:
    """Converts a string like 'QS', '10C', or '2H' into the 0-51 card index."""
    card_str = card_str.strip().upper()
    if len(card_str) < 2: return -1
    
    rank_char = card_str[:-1]
    suit_char = card_str[-1]
    
    suits = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    if suit_char not in suits: return -1
    suit_idx = suits[suit_char]
    
    ranks = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 
             'T':8, '10':8, 'J':9, 'Q':10, 'K':11, 'A':12}
    if rank_char not in ranks: return -1
    rank_idx = ranks[rank_char]
    
    return suit_idx * 13 + rank_idx


def print_game_state(env: GongZhuEnv, player_idx: int):
    """Renders the current trick and the human player's hand."""
    print("\n" + "="*50)
    print(f"--- TRICK {env.trick_num + 1} ---")
    
    if env.current_trick:
        played_str = " | ".join([f"P{p}: {card_name(c)}" for c, p in env.current_trick])
        print(f"Cards in play: {played_str}")
    else:
        print("You are leading the trick.")

    legal_actions = env.legal_actions(player_idx)
    print("\nYour hand (legal moves are numbered):")
    
    hand = sorted(list(env.hands[player_idx]))
    hand_display = []
    
    for c in hand:
        name = card_name(c)
        if c in legal_actions:
            idx = legal_actions.index(c)
            hand_display.append(f"[{idx}] {name}")
        else:
            hand_display.append(f"    {name} (illegal)")
            
    for i in range(0, len(hand_display), 5):
        print("  ".join(hand_display[i:i+5]))
    print("="*50)


def print_trick_winner(env: GongZhuEnv, winner: int, agents: list):
    """Helper to format and print the trick resolution."""
    last_trick = env.trick_history[-4:]
    scoring_cards = []
    for c, _, _, _ in last_trick:
        if c in CARD_BASE_SCORES:
            scoring_cards.append(f"{card_name(c)} ({CARD_BASE_SCORES[c]:+d})")
        elif c == TEN_CLUBS:
            scoring_cards.append(f"{card_name(c)} (x2 multiplier)")
    
    print(f"\n*** Player {winner} wins the trick! ***")
    if scoring_cards:
        print(f"    Picked up: {', '.join(scoring_cards)}")
    else:
        print(f"    Picked up: Clean trick (0 pts)")
        
    print("    Running Base Points & Cards:")
    for i in range(4):
        won_cards = env.tricks_won[i]
        pts = sum(CARD_BASE_SCORES.get(c, 0) for c in won_cards)
        mult = " [*10♣*]" if TEN_CLUBS in won_cards else ""
        
        scoring_ints = [c for c in won_cards if c in CARD_BASE_SCORES]
        banked_scoring = [card_name(c) for c in sorted(scoring_ints)]
        banked_str = f"  [{', '.join(banked_scoring)}]" if banked_scoring else ""
        
        winner_mark = "  <-- Won trick" if i == winner else ""
        print(f"      P{i}: {pts:+5d} pts{mult}{banked_str}{winner_mark}")
    
    if "human" in agents:
        time.sleep(1.5)


def get_action_with_probs(agent, env, print_probs=False):
    """Intercepts the PPOAgent to optionally print its internal EV and Action Probabilities."""
    # If it's a RandomAgent, it doesn't have a neural network
    if type(agent).__name__ == "RandomAgent":
        action = agent(env)
        if print_probs:
            print(f"   [Agent Probabilities] RandomAgent playing blindly.")
        return action

    # 1. Grab the lazy-loaded models
    ppo_model, belief_model = agent._models
    
    # 2. Build the exact observation the bot uses
    obs_np, snaps, legal_lists = agent._build_obs_batch([(env, env.current_player)])
    
    # 3. Run the Belief Network
    belief_channels = agent._run_batched_belief(belief_model, snaps, agent.device)
    obs_np[:, 7:10, :] = belief_channels.cpu().numpy()
    obs_t = torch.from_numpy(obs_np).to(agent.device)
    
    # 4. Build the legal action mask
    masks = np.zeros((1, 52), dtype=np.bool_)
    masks[0, legal_lists[0]] = True
    mask_t = torch.from_numpy(masks).to(agent.device)
    
    # 5. Extract probabilities and EV
    with torch.no_grad():
        logits = ppo_model.actor(obs_t)
        logits = logits.masked_fill(~mask_t, -1e8)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        value = ppo_model.critic(obs_t)[0].item()
        action = logits.argmax(dim=-1)[0].item()
    
    # 6. Format and print the output if requested
    if print_probs:
        from base_env import card_name
        legal_probs = [(c, probs[c]) for c in legal_lists[0]]
        legal_probs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   [Agent Probabilities] Underlying Expected Value (EV): {value:+.1f} pts")
        print("   [Agent Probabilities] Top Action Probabilities:")
        for i, (c, p) in enumerate(legal_probs[:3]):
            print(f"      {i+1}. {card_name(c):<4} ({p*100:5.1f}%)")
            
    return action


def run_simulation(agents: list, print_probs: bool = False):
    """Plays a fully simulated game where the environment deals the cards."""
    env = GongZhuEnv()
    env.reset()

    print("\nGame Start (Simulation Mode)!")
    print(f"Seats: P0:{agents[0]} | P1:{agents[1]} | P2:{agents[2]} | P3:{agents[3]}")

    while not env.done:
        curr_p = env.current_player
        agent = agents[curr_p]

        if agent == "human":
            print_game_state(env, curr_p)
            legal_actions = env.legal_actions(curr_p)
            while True:
                try:
                    choice = int(input(f"Player {curr_p} (Human), choose a card index [0-{len(legal_actions)-1}]: "))
                    if 0 <= choice < len(legal_actions):
                        action = legal_actions[choice]
                        break
                    print("Invalid index. Try again.")
                except ValueError:
                    print("Please enter a number.")
            print(f"-> You played {card_name(action)}")

        else:
            action = get_action_with_probs(agent, env, print_probs)
            if "human" in agents: time.sleep(0.8)
            agent_type = "PPO" if "PPO" in type(agent).__name__ else "Random"
            print(f"Player {curr_p} ({agent_type}) plays {card_name(action)}")

        env.step(action)

        if len(env.current_trick) == 0:
            winner = env.current_player 
            print_trick_winner(env, winner, agents)

    # Game Over Output
    print("\n" + "#"*50)
    print("GAME OVER")
    print("#"*50)
    scores = env.score()
    for p in range(4):
        agent_type = "Human" if agents[p] == "human" else type(agents[p]).__name__
        is_winner = " (WINNER)" if scores[p] == max(scores) else ""
        print(f"Player {p} ({agent_type:8s}): {scores[p]:5d} pts {is_winner}")
    print("#"*50)


def run_live_play(agents: list, print_probs: bool = False):
    """
    Live Physical Play: You distribute cards in the real world. 
    The script hijacks GongZhuEnv, using "Just-In-Time" card injection for human 
    players so the PPO agents can natively read the environment state.
    """
    print("\n" + "="*50)
    print(" LIVE PLAY MODE: You vs. The Bots")
    print("="*50)
    
    env = GongZhuEnv()
    env.reset() # Gives random hands, but we overwrite them below
    env.hands = [set() for _ in range(4)]
    env.done = False

    # Setup Bot Hands
    for p in range(4):
        if agents[p] != "human":
            print(f"\nEnter the 13 starting cards for Bot at Seat {p}.")
            print("Format: Rank+Suit (e.g., '2C 10H QS AS')")
            while True:
                hand_str = input(f"Bot {p} Hand: ")
                cards = [parse_card_string(s) for s in hand_str.split()]
                if len(cards) == 13 and all(c != -1 for c in cards):
                    env.hands[p] = set(cards)
                    break
                print("Invalid input. Ensure exactly 13 valid cards are entered.")
    
    env.current_player = int(input("\nWho leads the first trick? (0, 1, 2, or 3): "))
    
    while not env.done:
        curr_p = env.current_player
        agent = agents[curr_p]

        print(f"\n--- TRICK {env.trick_num + 1} | POS {len(env.current_trick) + 1}/4 ---")

        if agent != "human":
            action = get_action_with_probs(agent, env, print_probs)
            agent_type = "PPO" if "PPO" in type(agent).__name__ else "Random"
            print(f">>> BOT {curr_p} ({agent_type}) PLAYS: {card_name(action)} <<<")
        else:
            while True:
                card_str = input(f"What did Human Player {curr_p} play? ")
                action = parse_card_string(card_str)
                if action != -1:
                    if action in env.cards_in_play():
                        print("Error: That card has already been played!")
                        continue
                    # Magic Trick: Just-In-Time card injection. 
                    # We give the human the card right before they play it. 
                    # This satisfies GongZhuEnv's assertions and properly triggers void logic if they don't follow suit!
                    env.hands[curr_p].add(action)
                    break
                print("Invalid card format. Try again (e.g., 'JD').")
                
        env.step(action)
        
        if len(env.current_trick) == 0:
            winner = env.current_player
            print_trick_winner(env, winner, agents)
            
    print("\nGAME OVER! Hope the bot put up a good fight.")


def resolve_agent(agent_str: str, pool):
    """Maps CLI string to actual agent instance from the pool."""
    agent_str = agent_str.strip().lower()
    if agent_str == "human":
        return "human"
    if agent_str == "random":
        return pool.population[0] # Gen 0 is always Random
    if agent_str == "ppo":
        return pool.population[-1] # Latest PPO
    if agent_str.startswith("ppo_"):
        try:
            gen = int(agent_str.split("_")[1])
            if gen < len(pool.population):
                return pool.population[gen]
            else:
                print(f"Warning: Gen {gen} not found. Defaulting to latest.")
                return pool.population[-1]
        except ValueError:
            return pool.population[-1]
    if agent_str == "nash":
        if pool.distribution is None:
            print("Warning: No AlphaRank distribution found. Defaulting to latest PPO.")
            return pool.population[-1]
        chosen_idx = np.random.choice(len(pool.population), p=pool.distribution)
        gen_type = type(pool.population[chosen_idx]).__name__
        print(f"[Nash Selection] Bot secretly rolled Gen {chosen_idx} ({gen_type}) for this game.")
        return pool.population[chosen_idx]
    raise ValueError(f"Unknown agent type: {agent_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against GongZhu PSRO Bots")
    parser.add_argument("--save-dir", type=str, default="psro_runs", help="Directory containing pool_state.json")
    parser.add_argument("--players", type=str, default="human,ppo,ppo,ppo", 
                        help="Comma separated list of 4 players (human, random, ppo, ppo_1, etc.)")
    parser.add_argument("--print-probs", action="store_true", help="Print the bot's internal EV and action probabilities.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or "cpu"
    print(f"Loading PSRO Pool from '{args.save_dir}' on {device}...")

    try:
        pool, state = _load_pool(args.save_dir, device=device)
        gen = state.get("generation", 0)
        print(f"Successfully loaded Pool Generation: {gen} (Size: {len(pool)})")
    except FileNotFoundError:
        print(f"ERROR: Could not find pool state at {args.save_dir}.")
        print("Run `python run_psro_loop.py init` and train at least one generation first!")
        exit(1)

    player_strs = args.players.split(",")
    if len(player_strs) != 4:
        print("ERROR: --players must contain exactly 4 comma-separated values.")
        exit(1)

    # Resolve instances (PPOAgent, RandomAgent, or "human")
    resolved_agents = [resolve_agent(s, pool) for s in player_strs]

    mode = input("\nSelect Mode - [1] Simulation (Auto-dealt) or [2] Live Play (Physical Cards): ")
    if mode.strip() == "1":
        run_simulation(resolved_agents, args.print_probs)
    elif mode.strip() == "2":
        run_live_play(resolved_agents, args.print_probs)
    else:
        print("Invalid selection. Exiting.")
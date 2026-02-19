import torch
import numpy as np
import time
from typing import List
from foxzero.game import SevensGame, Card
from foxzero.model import FoxZeroNet
from foxzero.mcts import MCTS

class RandomAgent:
    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        return np.random.choice(valid_moves)

class FoxZeroAgent:
    def __init__(self, model_path: str, simulations=200):
        self.device = torch.device("cpu")
        self.net = FoxZeroNet().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        self.simulations = simulations

    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        
        # Use MCTS to select move
        # We need a new MCTS instance per move? 
        # Or reuse? Standard is new tree or subtree.
        # For simplicity, new tree.
        mcts = MCTS(self.net, simulations=self.simulations)
        
        # MCTS search returns probabilities
        pi = mcts.search(game)
        
        # Argmax selection for evaluation (Greedy)
        action_idx = np.argmax(pi)
        
        # Convert to Card
        s_idx = action_idx // 13
        r_idx = action_idx % 13
        card = Card(s_idx + 1, r_idx + 1)
        
        # Safety check
        if card not in valid_moves:
            # Fallback (shouldn't happen with MCTS masking)
            return valid_moves[0]
            
        return card

def evaluate(num_games=20):
    print(f"Evaluating FoxZero vs 3 Random Agents for {num_games} games...")
    
    # Load Model
    fox_agent = FoxZeroAgent("foxzero_model.pth", simulations=100)
    random_agent = RandomAgent()
    
    # FoxZero plays as Player 1
    # Note: In Sevens, Player 1 doesn't necessary start. Dealer starts.
    # We will track Player 1 (FoxZero) performance.
    
    fox_wins = 0
    fox_total_score_penalty = 0
    
    for i in range(num_games):
        game = SevensGame()
        
        # Game Loop
        while not game.is_game_over():
            current = game.current_player_number
            valid_moves = game.get_all_valid_moves(current)
            
            if not valid_moves:
                game.record_pass(current)
                game.next_player()
                continue
            
            if current == 1:
                # FoxZero (Player 1)
                card = fox_agent.select_move(game, current)
            else:
                # Random (Player 2, 3, 4)
                card = random_agent.select_move(game, current)
                
            game.make_move(card)
            game.next_player()
            
            # Simple infinite loop guard
            if game.turn_count > 300:
                print("Game aborted (too long).")
                break
        
        # Analyze Result
        rewards = game.calculate_final_rewards()
        # rewards are normalized [-1, 1].
        # Let's look at raw result.
        
        # Did P1 win?
        if game.has_player_won(1):
            fox_wins += 1
            result_str = "WIN"
        else:
            result_str = "LOSS"
            
        # Raw penalty for P1
        # Reconstruct or just peek hand
        p1_hand_score = sum(c.rank for c in game.hands[0].cards)
        fox_total_score_penalty += p1_hand_score
        
        print(f"Game {i+1}: FoxZero {result_str} (Left: {p1_hand_score} pts). Rewards: {rewards}")

    win_rate = fox_wins / num_games * 100
    avg_penalty = fox_total_score_penalty / num_games
    
    print("-" * 30)
    print(f"Evaluation Complete.")
    print(f"FoxZero Win Rate: {win_rate:.1f}%") # Chance is 25%
    print(f"Average Penalty Points: {avg_penalty:.1f}")
    
    if win_rate > 30.0:
        print("VERDICT: Model is performing better than random chance (25%).")
    else:
        print("VERDICT: Model performance is comparable to or worse than random.")

if __name__ == "__main__":
    evaluate()

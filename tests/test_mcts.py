import sys
import os
import numpy as np

# Ensure foxzero is in path
sys.path.append(os.getcwd())

from foxzero.game import SevensGame, Card, Suit
from foxzero.model import FoxZeroNet
from foxzero.mcts import MCTS

def test_mcts():
    print("Testing MCTS...")
    
    # 1. Setup
    game = SevensGame()
    net = FoxZeroNet() # Random weights
    mcts = MCTS(net, simulations=50) # Low sims for speed
    
    print(f"Current player: {game.current_player_number}")
    # Dealer (Spade 7 holder) starts.
    # We expect spade 7 to be valid. 
    # MCTS should return prob 1.0 for Spade 7 (Single Move Optimization).
    
    pi = mcts.search(game)
    
    valid_moves = game.get_all_valid_moves(game.current_player_number)
    print(f"Valid moves: {valid_moves}")
    
    if len(valid_moves) == 1:
        print("Single move detected. Expecting one-hot vector.")
        
        # Check if pi corresponds to valid move
        move = valid_moves[0]
        s_idx, r_idx = move.to_tensor_index()
        idx = s_idx * 13 + r_idx
        
        assert pi[idx] > 0.99
        print("Single move optimization passed.")
    
    # 2. Force a multi-choice scenario
    # Let's make a few moves to open up the board
    # Game sets dealer as Spade 7.
    # Let's verify MCTS can simulate.
    
    print("\nRunning MCTS from initial state (simulating play)...")
    # Actually initial state forces Spade 7.
    # Let's manually advance game until we have choices.
    
    # Dealer plays Spade 7
    spade_7 = Card(Suit.SPADE, 7)
    if game.is_valid_move(spade_7):
         game.make_move(spade_7)
         game.next_player()
         print("Played Spade 7.")
    
    # Now next player. Should have some moves?
    # Probably not guaranteed. 
    # But usually someone has Spade 6 or 8 or checks.
    # Let's just run search on whatever state.
    
    print(f"Now Player {game.current_player_number} turn.")
    valid = game.get_all_valid_moves(game.current_player_number)
    print(f"Valid moves: {valid}")
    
    if len(valid) > 1:
        print("Multiple moves available. Running MCTS...")
        pi = mcts.search(game)
        
        print(f"Pi sum: {np.sum(pi)}")
        assert np.isclose(np.sum(pi), 1.0)
        
        # Check non-zero probs are only on valid moves
        for i in range(52):
            if pi[i] > 0:
                # Recover card from index
                s_idx = i // 13
                r_idx = i % 13
                card = Card(s_idx + 1, r_idx + 1)
                
                assert card in valid
                
        print("MCTS probability distribution valid.")
    else:
        print("Still single move or pass. Skipping multi-choice test.")

    print("MCTS verification passed!")

if __name__ == "__main__":
    test_mcts()

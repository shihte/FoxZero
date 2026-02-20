import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foxzero.play import FoxZeroAgent
from foxzero.game import SevensGame

def test_speed():
    # Setup a game near the beginning
    game = SevensGame(4)
    # Give everyone a random hand automatically (setup_new_game does this)
    
    agent = FoxZeroAgent(model_path=None, simulations=800, c_puct=1.0, god_mode=False)
    
    print("Test MCTS benchmark - bypassing forced moves...")
    
    while True:
        current = game.current_player_number
        moves = game.get_all_valid_moves(current)
        if len(moves) > 1:
            break
            
        move = agent.select_move(game, current) 
        game.make_move(move)
        game.next_player()
    
    next_player = game.current_player_number
    
    print(f"Testing MCTS Inference speed with C++ Backend ({agent.simulations} sims) on Player {next_player}...")
    start = time.time()
    
    best_move = agent.select_move(game, next_player)
    
    end = time.time()
    
    print(f"Suggestion: {best_move}")
    print(f"Time Taken for {agent.simulations} simulations: {end - start:.2f} seconds")

if __name__ == '__main__':
    test_speed()

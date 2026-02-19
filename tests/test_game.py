import sys
from foxzero.game import SevensGame, Card, Suit, Rank

def test_game_logic():
    print("Testing FoxZero Game Logic...")
    game = SevensGame()
    
    # Verify initial state
    print(f"Num players: {game.num_players}")
    print(f"Dealer: {game.dealer_number}")
    print(f"Current Player: {game.current_player_number}")
    
    assert game.current_player_number == game.dealer_number
    
    dealer_hand = game.hands[game.dealer_number - 1]
    spade_7 = Card(Suit.SPADE, 7)
    assert spade_7 in dealer_hand.cards
    
    # Test valid moves
    valid_moves = game.get_all_valid_moves(game.current_player_number)
    print(f"Valid moves for dealer: {valid_moves}")
    assert spade_7 in valid_moves
    assert len(valid_moves) >= 1 # At least Spade 7
    
    # Test making a move
    print("Making move: Spade 7")
    game.make_move(spade_7)
    
    assert game.first_move_performed
    assert game.played_cards[Suit.SPADE - 1].lowest_card == spade_7
    assert game.played_cards[Suit.SPADE - 1].highest_card == spade_7
    
    # Verify next player
    expected_next = (game.dealer_number % 4) + 1
    game.next_player()
    assert game.current_player_number == expected_next
    
    # Test get_state_tensor
    try:
        import numpy as np
        print("Testing tensor generation...")
        tensor = game.get_state_tensor(game.current_player_number)
        print(f"Tensor shape: {tensor.shape}")
        assert tensor.shape == (11, 4, 13)
        print("Tensor generation successful.")
    except ImportError:
        print("Numpy not found, skipping tensor test.")
        
    print("Game logic verification passed!")

if __name__ == "__main__":
    test_game_logic()

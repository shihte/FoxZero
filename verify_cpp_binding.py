import sys
import os

# Add current directory to path so we can import sevens_core (it's an .so file here)
sys.path.append(os.getcwd())

try:
    import sevens_core
    print("SUCCESS: sevens_core imported successfully.")
except ImportError as e:
    print(f"FAILURE: Could not import sevens_core. {e}")
    sys.exit(1)

def test_engine():
    print("Testing SevensEngine...")
    engine = sevens_core.SevensEngine()
    engine.setupNewGame()
    
    print(f"Number of players: {engine.getHands().__len__()}") # Accessing size of vector
    # Hands is vector<Hand>, pybind11 exposes size() as __len__ usually if bound as opaque?
    # No, I didn't bind vector container details explicitly, but pybind11/stl.h handles it.
    
    hands = engine.getHands()
    print(f"Hands count: {len(hands)}")
    
    p = engine.getCurrentPlayerNumber()
    print(f"Current Player: {p}")
    
    obs = engine.get_observation(p)
    print(f"Observation size: {len(obs)}")
    if len(obs) != 11 * 4 * 13:
        print(f"FAILURE: Observation size incorrect. Expected {11*4*13}, got {len(obs)}")
        sys.exit(1)
        
    print("Observation generation successful.")
    
    valid_moves = engine.getAllValidMoves(p)
    print(f"Valid moves for player {p}: {[c.__repr__() for c in valid_moves]}")
    
    legal_mask = engine.get_legal_moves()
    print(f"Legal mask sum: {sum(legal_mask)}")
    
    # Try step
    if valid_moves:
        c = valid_moves[0]
        idx = (c.suit - 1) * 13 + (c.rank - 1)
        print(f"Making move: {c.__repr__()} (idx {idx})")
        engine.step(idx)
        print("Move successful.")
        print(f"New Current Player: {engine.getCurrentPlayerNumber()}")
    else:
        print("No valid moves (Cover needed? Should be handled by getAllValidMoves return all cards if so)")
        
    print("SevensEngine test PASSED.")

if __name__ == "__main__":
    test_engine()

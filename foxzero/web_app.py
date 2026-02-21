from flask import Flask, render_template, request, jsonify
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foxzero.analysis import run_analysis
from foxzero.game import SevensGame, Card
from foxzero.play import FoxZeroAgent

app = Flask(__name__, template_folder='templates')

# --- Web Game Global State ---
game_session = None
global_ai_agent = None

def get_ai_agent():
    global global_ai_agent
    if global_ai_agent is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "foxzero_weights.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "foxzero_model.pth")
        global_ai_agent = FoxZeroAgent(model_path=model_path, simulations=400, c_puct=1.0)
    return global_ai_agent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play')
def play():
    return render_template('play.html')

# --- API Endpoints ---
@app.route('/api/game/start', methods=['POST'])
def api_start_game():
    global game_session
    game_session = SevensGame()
    # Preload the AI model async or just trigger it
    get_ai_agent()
    return jsonify({"success": True, "message": "Game started"})

@app.route('/api/game/state', methods=['GET'])
def api_game_state():
    global game_session
    if not game_session:
        return jsonify({"error": "No active game."})
        
    is_over = game_session.is_game_over()
    curr_player = game_session.current_player_number
    
    # Format Player 1 hand
    p1_hand = [{"suit": c.suit, "rank": c.rank} for c in sorted(game_session.hands[0].cards, key=lambda x: (x.suit, x.rank))]
    
    # Format Board
    board = {}
    for s in range(1, 5):
        ps = game_session.played_cards[s - 1]
        board[str(s)] = {
            "lowest": ps.lowest_card.rank if ps.lowest_card else None,
            "highest": ps.highest_card.rank if ps.highest_card else None
        }
        
    # Valid moves for current player
    valid_moves_objs = game_session.get_all_valid_moves(curr_player)
    valid_moves = [{"suit": c.suit, "rank": c.rank, "is_cover": not game_session.is_valid_move(c)} for c in valid_moves_objs]
    
    # Opponent statuses
    opponents = {}
    for p in range(2, 5):
        opponents[str(p)] = {
            "hand_count": len(game_session.hands[p - 1].cards),
            "covered_count": len(game_session.covered_cards[p - 1])
        }
        
    # Player 1 covered points
    p1_covered = [c.rank for c in game_session.covered_cards[0]]
    
    response = {
        "turn_count": game_session.turn_count,
        "current_player": curr_player,
        "is_game_over": is_over,
        "p1_hand": p1_hand,
        "p1_covered": sum(p1_covered),
        "board": board,
        "valid_moves": valid_moves,
        "opponents": opponents
    }
    
    # If game over, calculate final rewards
    if is_over:
        rewards = game_session.calculateFinalRewards()
        response["rewards"] = rewards
        
    return jsonify(response)

@app.route('/api/game/make_move', methods=['POST'])
def api_make_move():
    global game_session
    if not game_session:
        return jsonify({"error": "No active game."})
        
    data = request.json
    curr = game_session.current_player_number
    
    if curr != 1:
        return jsonify({"error": "Not your turn!"})
        
    if data and "suit" in data and "rank" in data:
        card = Card(data['suit'], data['rank'])
        is_cover = not game_session.is_valid_move(card)
        game_session.make_move(card)
        game_session.next_player()
        return jsonify({"success": True, "move": {"suit": card.suit, "rank": card.rank, "is_cover": is_cover}})
    else:
        # Pass (should not happen in Sevens)
        return jsonify({"success": False, "error": "Invalid move"})

@app.route('/api/game/ai_move', methods=['POST'])
def api_ai_move():
    global game_session
    if not game_session:
        return jsonify({"error": "No active game."})
        
    curr = game_session.current_player_number
    if curr == 1:
        return jsonify({"error": "It is human's turn!"})
        
    agent = get_ai_agent()
    move = agent.select_move(game_session, curr)
    
    if move:
        is_cover = not game_session.is_valid_move(move)
        game_session.make_move(move)
        game_session.next_player()
        return jsonify({"success": True, "move": {"suit": move.suit, "rank": move.rank, "is_cover": is_cover}})
    else:
        return jsonify({"success": False, "move": None})

# --- Analysis Endpoint ---

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    # Extract data
    ranges = data.get('ranges', {})
    s_r = ranges.get('s', [])
    h_r = ranges.get('h', [])
    c_r = ranges.get('c', [])
    d_r = ranges.get('d', [])
    
    covered = data.get('covered', {})
    cp1 = covered.get('p1', 0)
    cp2 = covered.get('p2', 0)
    cp3 = covered.get('p3', 0)
    cp4 = covered.get('p4', 0)
    
    hand_input = data.get('hand', []) # List of strings "S1", "H13" etc.
    if isinstance(hand_input, list):
        hand_input = ", ".join(hand_input)
        
    sims = data.get('simulations', 500)
    
    try:
        result = run_analysis(
            s_r, h_r, c_r, d_r,
            cp1, cp2, cp3, cp4,
            hand_input,
            sims
        )
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result)
        return jsonify({'output': result})
    except Exception as e:
        return jsonify({'error': str(e), 'output': ''})

if __name__ == '__main__':
    print("🦊 FoxZero Web Analysis Tool Starting...")
    print("🌍 Open: http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)

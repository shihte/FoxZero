from flask import Flask, render_template, request, jsonify
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foxzero.analysis import run_analysis

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

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

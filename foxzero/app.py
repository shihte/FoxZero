import gradio as gr
import torch
import numpy as np
import copy
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foxzero.game import SevensGame, Card, Suit, Rank
from foxzero.play import run_mcts_inference, FoxZeroResNet

# --- Logic ---

def card_name(suit, rank):
    suits = {1: "♦️", 2: "♣️", 3: "❤️", 4: "♠️"}
    ranks = {1: "A", 11: "J", 12: "Q", 13: "K"}
    r_str = ranks.get(rank, str(rank))
    return f"{suits[suit]}{r_str}"

def parse_hand(hand_str):
    """Parses a comma-separated list of cards (e.g. 'S7, H1, D13', '♠️1', '♥12') into Card objects."""
    cards = []
    # Normalize input: uppercase check only for standard letters
    parts = hand_str.split(',')
    
    suit_map_char = {'S': 4, 'H': 3, 'C': 2, 'D': 1}
    # Rank map for non-digits
    rank_map_str = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    
    for p in parts:
        p = p.strip()
        if not p: continue
        
        # Identify Suit
        found_suit = None
        
        # Check standard chars
        p_upper = p.upper()
        for k, v in suit_map_char.items():
            if k in p_upper:
                found_suit = v
                break
        
        if not found_suit:
            # Check symbols
            if '♠' in p: found_suit = 4
            elif '♥' in p: found_suit = 3
            elif '♣' in p: found_suit = 2
            elif '♦' in p: found_suit = 1
            
        if not found_suit:
            continue # Invalid
            
        # Extract Rank
        # Remove suit symbols and standard suit chars, then parse what remains
        # Safe way: extract digits or Letters A,J,Q,K
        
        # Remove known suit chars/symbols to clean up p
        clean_p = p.upper().replace('♠', '').replace('♥', '').replace('♣', '').replace('♦', '')
        # Remove S,H,D,C if they were used as suit indicators? 
        # But 'S' might be part of 'Six'? No.
        for k in suit_map_char:
             clean_p = clean_p.replace(k, '')
             
        # Remove variation selectors \ufe0f
        clean_p = clean_p.replace('\ufe0f', '').strip()
        
        # Now clean_p should be '7', '13', 'K', '10' etc.
        rank = 0
        if clean_p in rank_map_str:
            rank = rank_map_str[clean_p]
        else:
            try:
                rank = int(clean_p)
            except:
                pass
                
        if 1 <= rank <= 13:
            cards.append(Card(found_suit, rank))
            
    return cards

def run_analysis(spade_range, heart_range, club_range, diamond_range, 
                 covered_p1, covered_p2, covered_p3, covered_p4,
                 my_hand_input, simulations):
    
    # 1. Setup Game State
    game = SevensGame()
    
    # Set Played Cards (Board)
    # Range is [min, max]. e.g. [4, 10].
    # If standard start (7), cards played are 7..max and min..7.
    # But Sevens grows 7->6->... and 7->8->...
    # So if range is [4, 10], it implies 4,5,6,7,8,9,10 are played.
    # BUT, 7 must be played to play 6 or 8. So it's always contiguous from 7.
    # Unless 7 is NOT played? Sevens game starts with S7.
    # Assuming valid state: contiguous range including 7 (if played) or single 7?
    # Actually, if range is [4, 10], it means 4,5,6,7,8,9,10 are on board.
    
    # Reset played_cards
    # SevensGame init sets played_cards to just placeholders (1,2,3,4) with empty ranges?
    # No, game.played_cards is List[PlacedSuit].
    # PlacedSuit tracks low/high.
    
    # We need to manually set `low` and `high` for each suit.
    # S:1, H:2, C:3, D:4
    ranges = {
        4: spade_range, 
        3: heart_range, 
        2: club_range, 
        1: diamond_range
    }
    
    # Validate ranges contain 7?
    # In Sevens, you can't play 6 unless 7 is out.
    # So range MUST include 7 if any card is played?
    # Exception: If Suit 7 is NOT played, flow hasn't started for that suit.
    # But RangeSlider forces a range.
    # If user selects [7, 7], implies 7 is out.
    # What if suit not started?
    # We need a way to say "None".
    # But RangeSlider always has value.
    # Maybe logic: If range == [7, 7] and user says "Not Started"?
    # Let's assume User smart: If range is [7, 7], 7 is played.
    # How to represent "Nothing played"?
    # Sevens mechanics: S7 starts. Others optional.
    # Maybe add "Has 7 been played?" checkbox per suit?
    # Or simpler: Range 0-0 means empty? No, RangeSlider min/max.
    # Let's stick to: Range [x, y] means x..y played.
    # If [7, 7], 7 is played.
    
    total_played = 0
    
    for suit_id, (low, high) in ranges.items():
        # Update game state
        # PlacedSuit: has_7, low, high
        # If range includes 7, set has_7=True.
        # But wait, logic: 7 is ALWAYS first.
        # So if range is [4, 10], 7 MUST be in there.
        # If range doesn't include 7 (e.g. [1, 3]), it's INVALID state.
        # We will assume valid input: 7 is inside or range is empty?
        # RangeSlider constraints?
        
        # Let's trust user input but fix logically.
        # If 7 not in range, we force it? Or assume invalid?
        # Let's assume user sets valid ranges (containing 7).
        
        ps = game.played_cards[suit_id - 1]
        ps.has_7 = True # Assume played for now
        ps.low = low
        ps.high = high
        
        # Calculate count
        count = high - low + 1
        total_played += count

    # 2. Setup Hands
    # We have `my_hand_input` (list of cards).
    # We have `covered_counts` for p1..p4.
    
    # We need to assign `my_hand`.
    # Who am I? Let's assume Player 1 for now.
    user_player_id = 1
    
    my_cards = parse_hand(my_hand_input) if isinstance(my_hand_input, str) else [] # CheckboxGroup handling needed?
    # If CheckboxGroup, it returns list of strings.
    if isinstance(my_hand_input, list):
         # e.g. ["S7", "H1"]
         my_cards = parse_hand(", ".join(my_hand_input))
         
    game.hands[user_player_id - 1].cards = my_cards
    
    # 3. Determine Current Player
    # Total history = total_played_on_board + total_covered
    total_covered = covered_p1 + covered_p2 + covered_p3 + covered_p4
    # Wait, total_played calculation above included ALL cards in range.
    # But S7 starts.
    # Turn 0: S7 played. Total=1. Next P2.
    # So `turn_count` = total_cards_out - 1?
    # `SevensGame` starts with empty board? No.
    # `game.played_cards` init: has_7=False.
    # If we set ranges, we reflect "Turn X".
    # Total cards "dealt with" (played or covered) = `total_played + total_covered`.
    # But wait, board includes S7?
    # If S7 is on board, it counts as played.
    # If S7 is NOT on board (impossible?), count=0.
    
    total_actions = total_played + total_covered
    
    # S7 is usually first action.
    # If S7 on board, it counts as 1 action.
    # So `turn_count` = total_actions.
    # Except S7 is often considered "Turn 0" or automated?
    # In my `SevensGame`, `make_move` increments turn.
    # If S7 is played, turn becomes 1. P2 moves.
    # So `current_player = (turn_count % 4) + 1`.
    
    # But there's an offset.
    # Who starts? Holder of S7 (Player 1 usually re-assigned or random).
    # Since we analyze Arbitrary State, we assume Player 1 is "Me".
    # Who moved to get here?
    # We assume standard order 1->2->3->4.
    # S7 owner is Player 1?
    # If user is P1, and S7 is on board...
    # Did P1 play S7?
    # If yes, and P1 is Dealer (1st player).
    
    # Let's assume simple rotation:
    # `current_player = (total_actions % 4) + 1`?
    # If total_actions=1 (S7 played), next is P2. Correct.
    # If total_actions=0 (Nothing played), next is P? (Start of game).
    
    # Correction: `turn_number` starts at 1?
    # `game.current_player_number` is 1..4.
    
    # We set `game.turn_count = total_actions`.
    # `game.current_player_number = (game.dealer_number + total_actions - 1) % 4 + 1`?
    # Default dealer=1.
    # If total=0, Current=1.
    # If total=1 (S7 out), Current=1 + 1 = 2.
    # Formula: `(1 + total_actions - 1) % 4 + 1` -> `total_actions % 4 + 1`.
    # Seems correct.
    
    game.turn_count = total_actions
    game.current_player_number = (total_actions % 4) + 1
    
    # 4. Infer Forbidden Cards (Constraints)
    # We can't know for sure. Logic Mode uses `forbidden_cards` set during play.
    # Here we are "jumping in".
    # We can assume that if a player covered, they didn't have Playable cards.
    # BUT we don't know WHEN they covered.
    # So we can't easily populate `forbidden_cards`.
    # We will skip strict constraints for now, or just rely on `determinize` to fill hands validly.
    
    # 5. Determinize (Fill Opponent Hands)
    # logic mode active?
    # We need to fill `covered_cards` piles for opponents too.
    # My `determinize` only fills HANDS.
    # We need to manually remove `covered_counts` cards from the "Unknown Pool" 
    # and put them in `game.covered_cards`.
    
    # Identify Unknown Cards
    all_cards = set()
    for s in range(1, 5):
        for r in range(1, 14):
            all_cards.add(Card(s, r))
            
    # Remove Board Cards
    for suit_id, (low, high) in ranges.items():
        # Assumes contiguous range exists
        if low <= high: # Valid range
            # Check if this suit is "started"
            # If S7 is required, check if 7 in range.
            pass
        for r in range(low, high + 1):
             c = Card(suit_id, r)
             if c in all_cards: all_cards.remove(c)
             
    # Remove User Hand
    for c in my_cards:
        if c in all_cards: all_cards.remove(c)
        
    # Remaining are Unknowns (Opponent Hands + Opponent Covered)
    unknowns = list(all_cards)
    
    # Shuffle
    import random
    random.shuffle(unknowns)
    
    # Distribute to Covered Piles & Hands considering Counts
    # We need:
    # P1: Hand Known (my_cards). Covered: `covered_p1`.
    # P2: Hand ?, Covered `covered_p2`.
    # P3: Hand ?, Covered `covered_p3`.
    # P4: Hand ?, Covered `covered_p4`.
    
    # Assign Covered Cards first (from unknowns)
    # Note: Logic Mode uses `forbidden_cards` to guide this. Here we just random.
    current_idx = 0
    
    player_configs = [
        (1, covered_p1),
        (2, covered_p2),
        (3, covered_p3),
        (4, covered_p4)
    ]
    
    for pid, cov_count in player_configs:
        start = current_idx
        end = current_idx + cov_count
        if end > len(unknowns):
             # Error in counting?
             return f"錯誤：牌數不符。剩餘未知牌 {len(unknowns)} 張，但需要分配蓋牌+手牌。", None
             
        p_covered = unknowns[start:end]
        game.covered_cards[pid - 1] = p_covered
        current_idx = end
        
    # Distribute Remaining to Hands
    # How many cards should each player have?
    # Total 13 per player.
    # Played cards belong to someone.
    # We don't know who played what on board.
    # This is a problem for `determinize` if it expects hand sizes to match history?
    # `SevensGame.determinize` uses `self.hands[i].get_card_count()` to know how many to deal.
    # But here empty hands.
    # We need to infer "Remaining Hand Size".
    # Remaining Hand Size = 13 - (Times Moved).
    # Times Moved approx = Turn Count / 4.
    # Player P moves at Turn P, P+4, P+8...
    # Let's calculate exact remaining count.
    
    for pid in range(1, 5):
        # Calculate moves made
        moves_made = 0
        # If current turn is T. (0-indexed).
        # P1 moves at 0, 4, 8...
        # If T=0 (P1 to move), P1 made 0 moves.
        # If T=1 (P2 to move), P1 made 1 move.
        
        # moves = (total_actions + (4 - (pid - 1)) - 1) // 4 ? Complex.
        # Simpler:
        # Loop t from 0 to total_actions - 1.
        # mover = (t % 4) + 1.
        # if mover == pid: moves_made += 1
        
        moves_made = 0
        for t in range(total_actions):
            mover = (t % 4) + 1
            if mover == pid: moves_made += 1
            
        remaining_hand = 13 - moves_made
        
        if pid == user_player_id:
            # Check user input matches?
            if len(my_cards) != remaining_hand:
                # Warning or Error?
                # Maybe user input implies they covered some? 
                # User config has `covered_p1`.
                # If I covered X cards and played Y cards (inferred from moves_made - X), then remaining = 13 - moves_made.
                pass
        else:
            # Assign random cards to opponent hand
            needed = remaining_hand
            start = current_idx
            end = current_idx + needed
            if end > len(unknowns):
                 # Error
                 pass
            
            p_hand = unknowns[start:end]
            game.hands[pid-1].cards = p_hand
            current_idx = end
            
    # 6. Run MCTS Inference
    # Load Model (Global or Lazy Load)
    
    # Current Player logic check
    curr = game.current_player_number
    
    # Output State
    info = f"當前玩家: {curr}\n"
    info += f"回合數: {game.turn_count}\n"
    info += f"桌面: ♠{spade_range} ♥{heart_range} ♣{club_range} ♦{diamond_range}\n"
    info += f"推演次數: {simulations}\n"
    
    # Run
    # Setup model
    model = FoxZeroResNet()
    model_path = "models/foxzero_weights.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Capture stdout
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        best_card = run_mcts_inference(
            game, 
            model, 
            simulations=simulations, 
            god_mode=False # Logic Mode
        )
    except Exception as e:
        sys.stdout = old_stdout
        return f"錯誤：{e}\n{info}", ""
        
    sys.stdout = old_stdout
    log_output = mystdout.getvalue()
    
    result_str = f"✅ 建議出牌: {best_card}\n\n{info}\n\n--- 思考日誌 ---\n{log_output}"
    return result_str

# --- UI ---

with gr.Blocks(title="FoxZero 算牌器") as demo:
    gr.Markdown("# 🦊 FoxZero 邏輯推演算牌器")
    gr.Markdown("設定當前牌局狀態，讓 AI 幫你推算最佳出牌！")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🃏 桌面狀態 (已出的牌)")
            gr.Markdown("請設定每種花色目前桌面上的 **最小點數** 和 **最大點數** (包含已出的 7)。若尚未出牌，請設為 7-7 (代表僅 7 可出/已出)。")
            
            with gr.Row():
                s_min = gr.Slider(1, 13, value=7, step=1, label="♠️ 黑桃 Min")
                s_max = gr.Slider(1, 13, value=7, step=1, label="♠️ 黑桃 Max")
            with gr.Row():
                h_min = gr.Slider(1, 13, value=7, step=1, label="❤️ 紅心 Min")
                h_max = gr.Slider(1, 13, value=7, step=1, label="❤️ 紅心 Max")
            with gr.Row():
                c_min = gr.Slider(1, 13, value=7, step=1, label="♣️ 梅花 Min")
                c_max = gr.Slider(1, 13, value=7, step=1, label="♣️ 梅花 Max")
            with gr.Row():
                d_min = gr.Slider(1, 13, value=7, step=1, label="♦️ 方塊 Min")
                d_max = gr.Slider(1, 13, value=7, step=1, label="♦️ 方塊 Max")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🙈 玩家蓋牌數")
            cov_p1 = gr.Number(value=0, label="玩家 1 (我) 蓋牌", minimum=0, maximum=13)
            cov_p2 = gr.Number(value=0, label="玩家 2 蓋牌", minimum=0, maximum=13)
            cov_p3 = gr.Number(value=0, label="玩家 3 蓋牌", minimum=0, maximum=13)
            cov_p4 = gr.Number(value=0, label="玩家 4 蓋牌", minimum=0, maximum=13)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🖐️ 我的手牌")
            # All 52 cards checkbox is too big.
            # Using Text Input for compactness or 4 Parsed Groups.
            # Let's use 4 CheckboxGroups by suit.
            
            def get_suit_cards(suit_icon):
                return [f"{suit_icon}{r}" for r in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
                
            # Parse back logic needed.
            # Actually, CheckboxGroup returns list of strings "♠️1".
            # Easy to parse.
            
            chk_s = gr.CheckboxGroup(choices=get_suit_cards("♠️"), label="黑桃手牌")
            chk_h = gr.CheckboxGroup(choices=get_suit_cards("❤️"), label="紅心手牌")
            chk_c = gr.CheckboxGroup(choices=get_suit_cards("♣️"), label="梅花手牌")
            chk_d = gr.CheckboxGroup(choices=get_suit_cards("♦️"), label="方塊手牌")
            
    with gr.Row():
        sims_slider = gr.Slider(0, 5000, value=500, step=100, label="🧠 推演次數 (Simulations)")
        
    btn = gr.Button("🚀 開始推演 (Calculate Best Move)", variant="primary")
    output = gr.Textbox(label="推演結果", lines=10)
    
    def process_inputs(s_min, s_max, h_min, h_max, c_min, c_max, d_min, d_max, cp1, cp2, cp3, cp4, h_s, h_h, h_c, h_d, sims):
        # Combine hands
        full_hand_strs = h_s + h_h + h_c + h_d
        # Format for parser: "♠️1, ❤️13..."
        hand_str = ", ".join(full_hand_strs)
        
        # Combine ranges
        # Ensure min <= max
        s_r = [min(s_min, s_max), max(s_min, s_max)]
        h_r = [min(h_min, h_max), max(h_min, h_max)]
        c_r = [min(c_min, c_max), max(c_min, c_max)]
        d_r = [min(d_min, d_max), max(d_min, d_max)]
        
        return run_analysis(s_r, h_r, c_r, d_r, 
                            int(cp1), int(cp2), int(cp3), int(cp4), 
                            hand_str, int(sims))

    btn.click(
        process_inputs,
        inputs=[s_min, s_max, h_min, h_max, c_min, c_max, d_min, d_max, cov_p1, cov_p2, cov_p3, cov_p4, chk_s, chk_h, chk_c, chk_d, sims_slider],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False)

import torch
import numpy as np
import copy
import sys
import os
from io import StringIO

# Add project root to path if needed (logic handles relative imports usually)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foxzero.game import SevensGame, Card, Suit, Rank
from foxzero.play import run_mcts_inference, FoxZeroResNet

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
        # Remove S,H,D,C if they were used as suit indicators
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
    
    # We need to manually set `low` and `high` for each suit.
    # S:1, H:2, C:3, D:4
    ranges = {
        4: spade_range, 
        3: heart_range, 
        2: club_range, 
        1: diamond_range
    }
    
    total_played = 0
    
    for suit_id, (low, high) in ranges.items():
        # Update game state
        ps = game.played_cards[suit_id - 1]
        
        if low == 0 or high == 0:
            ps.lowest_card = None
            ps.highest_card = None
            count = 0
        else:
            ps.lowest_card = Card(suit_id, low)
            ps.highest_card = Card(suit_id, high)
            # Calculate count
            count = high - low + 1
            
        total_played += count

    # 2. Setup Hands
    # We have `my_hand_input` (list of cards).
    # We have `covered_counts` for p1..p4.
    
    # We need to assign `my_hand`.
    # Who am I? Let's assume Player 1 by default.
    user_player_id = 1
    
    my_cards = parse_hand(my_hand_input) if isinstance(my_hand_input, str) else [] 
    if isinstance(my_hand_input, list):
         my_cards = parse_hand(", ".join(my_hand_input))
         
    game.hands[user_player_id - 1].cards = my_cards
    
    # 3. Determine Current Player
    # Total history = total_played_on_board + total_covered
    total_covered = covered_p1 + covered_p2 + covered_p3 + covered_p4
    
    total_actions = total_played + total_covered
    
    # S7 is usually first action.
    # If S7 on board, it counts as 1 action. (Assuming standard deal where S7 forces play)
    # Formula for current player: (total_actions % 4) + 1
    # If total=0 (start), P1.
    # If total=1 (S7 played), P2.
    
    if total_played > 0:
        game.first_move_performed = True
        
    game.turn_count = total_actions
    game.current_player_number = 1
    
    # Calculate moves made by each player recursively assuming it's currently P1's turn
    # MCTS requires valid hand counts.
    moves_made = {1: 0, 2: 0, 3: 0, 4: 0}
    base_moves = total_actions // 4
    rem = total_actions % 4
    for p in range(1, 5):
        moves_made[p] = base_moves
    # The 'rem' players who played before P1 (P4, P3, P2 backwards)
    if rem >= 1: moves_made[4] += 1
    if rem >= 2: moves_made[3] += 1
    if rem >= 3: moves_made[2] += 1

    # 4. Infer Unknowns & Redistribute
    
    # Identify Unknown Cards
    all_cards = set()
    for s in range(1, 5):
        for r in range(1, 14):
            all_cards.add(Card(s, r))
            
    # Remove Board Cards
    for suit_id, (low, high) in ranges.items():
        if low <= high: 
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
    
    current_idx = 0
    
    player_configs = [
        (1, covered_p1),
        (2, covered_p2),
        (3, covered_p3),
        (4, covered_p4)
    ]
    
    # Assign Covered Cards
    for pid, cov_count in player_configs:
        start = current_idx
        end = current_idx + cov_count
        if end > len(unknowns):
             return f"錯誤：牌數不符。剩餘未知牌 {len(unknowns)} 張，但需要分配更多蓋牌。", ""
             
        p_covered = unknowns[start:end]
        game.covered_cards[pid - 1] = p_covered
        current_idx = end
        
    # Assign Hands
    for pid in range(1, 5):
        remaining_hand = 13 - moves_made[pid]
        
        if pid == user_player_id:
            needed = remaining_hand - len(my_cards)
            if needed < 0:
                 return f"❌ 錯誤：手牌選太多了！\n根據桌面已出牌數 ({total_played} 張)，您的手牌應該剩下 {remaining_hand} 張，但您勾選了 {len(my_cards)} 張。\n請取消勾選一些牌。", ""
                 
            # If they checked fewer cards, fill the rest with random unknowns
            start = current_idx
            end = current_idx + needed
            if end > len(unknowns):
                  return f"❌ 錯誤：牌數邏輯錯誤。系統沒有足夠的未知牌發給您。", ""
            
            game.hands[pid-1].cards = my_cards + unknowns[start:end]
            current_idx = end
        else:
            # Assign random cards to opponent hand
            needed = remaining_hand
            start = current_idx
            end = current_idx + needed
            if end > len(unknowns):
                  return f"❌ 錯誤：牌數邏輯錯誤。剩餘未知牌不足以分配給對手，請檢查桌面狀態和蓋牌設定。", ""
            
            p_hand = unknowns[start:end]
            game.hands[pid-1].cards = p_hand
            current_idx = end
            
    # 6. Run MCTS Inference
    # Load Model (Global or Lazy Load)
    
    # Current Player logic check
    curr = game.current_player_number
    
    # Output State Info
    info = f"當前玩家: {curr}\n"
    info += f"回合數: {game.turn_count}\n"
    info += f"桌面: ♠{spade_range} ♥{heart_range} ♣{club_range} ♦{diamond_range}\n"
    info += f"推演次數: {simulations}\n"
    
    # Run
    # Setup model
    model = FoxZeroResNet()
    model_path = "foxzero_weights.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Capture stdout
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
        import traceback
        return f"錯誤：{e}\n{traceback.format_exc()}\n{info}", ""
        
    sys.stdout = old_stdout
    log_output = mystdout.getvalue()
    
    if game.is_valid_move(best_card):
        action_text = f"✅ 建議出牌: {best_card}"
    else:
        action_text = f"⚠️ 建議蓋牌 (被迫): {best_card}"
        
    result_str = f"{action_text}\n\n{info}\n\n--- 思考日誌 ---\n{log_output}"
    return result_str

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
    
    for suit_id, r in ranges.items():
        # Update game state
        ps = game.played_cards[suit_id - 1]
        
        if not r or len(r) != 2:
            ps.lowest_card = None
            ps.highest_card = None
            count = 0
            ps.has_7 = False
        else:
            low, high = r
            if low == 0 or high == 0:
                ps.lowest_card = None
                ps.highest_card = None
                count = 0
                ps.has_7 = False
            else:
                ps.lowest_card = Card(suit_id, low)
                ps.highest_card = Card(suit_id, high)
                ps.has_7 = True
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
    
    # 3. Determine Game State from User Hand
    total_ui_actions = total_played + covered_p1 + covered_p2 + covered_p3 + covered_p4
    user_moves = 13 - len(my_cards)
    
    if total_played > 0:
        game.first_move_performed = True
        
    game.current_player_number = 1
    moves_made = {1: user_moves, 2: user_moves, 3: user_moves, 4: user_moves}
    
    # Auto-balance missing actions using P1's absolute turn truth
    if total_ui_actions <= 4 * user_moves:
        missing_covers = 4 * user_moves - total_ui_actions
        # Add casually omitted covers to opponents to balance the timeline
        for i in range(missing_covers):
            if i % 3 == 0: covered_p2 += 1
            elif i % 3 == 1: covered_p3 += 1
            else: covered_p4 += 1
        total_actions = 4 * user_moves
    else:
        k = total_ui_actions - 4 * user_moves
        if k > 3:
            return {"error": f"❌ 設定矛盾！您目前手牌剩下 {len(my_cards)} 張 (出了 {user_moves} 次)，但桌面上的牌 + 蓋牌居然有 {total_ui_actions} 張！這代表其他人出牌次數遠超過正常範圍，請檢查有無選錯桌面花色範圍或蓋牌數量。"}
            
        if k >= 1: moves_made[4] += 1
        if k >= 2: moves_made[3] += 1
        if k >= 3: moves_made[2] += 1
        total_actions = total_ui_actions

    game.turn_count = total_actions

    # 4. Infer Unknowns & Redistribute
    
    # Identify Unknown Cards
    all_cards = set()
    for s in range(1, 5):
        for r in range(1, 14):
            all_cards.add(Card(s, r))
            
    # Remove Board Cards
    for suit_id, r in ranges.items():
        if r and len(r) == 2:
            low, high = r
            if low <= high and low > 0: 
                for r_val in range(low, high + 1):
                     c = Card(suit_id, r_val)
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
             return {"error": f"錯誤：牌數不符。剩餘未知牌 {len(unknowns)} 張，但需要分配更多蓋牌。"}
             
        p_covered = unknowns[start:end]
        game.covered_cards[pid - 1] = p_covered
        current_idx = end
        
    # Assign Hands
    for pid in range(1, 5):
        remaining_hand = 13 - moves_made[pid]
        
        if pid == user_player_id:
            # We've guaranteed moves_made[1] matches len(my_cards) exactly
            game.hands[pid-1].cards = my_cards
        else:
            # Assign random cards to opponent hand
            needed = remaining_hand
            start = current_idx
            end = current_idx + needed
            if end > len(unknowns):
                  return {"error": f"❌ 錯誤：牌數邏輯錯誤。剩餘未知牌不足以分配給對手，這通常代表「桌面出牌」與「您的手牌數」有矛盾。請檢查！"}
            
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
    model_path = "models/foxzero_weights.pth"
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
        return {"error": f"推演發生錯誤：{e}\n{traceback.format_exc()}\n{info}"}
        
    sys.stdout = old_stdout
    log_output = mystdout.getvalue()
    
    is_play = game.is_valid_move(best_card)
    
    suit_map = {1: 'd', 2: 'c', 3: 'h', 4: 's'}
    
    return {
        "action": str(best_card),
        "suit": suit_map[best_card.suit],
        "rank": best_card.rank,
        "is_play": is_play,
        "info": info,
        "log": log_output
    }

import torch
import numpy as np
import time
import argparse
import sys
import os
import math
import copy

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foxzero.game import SevensGame, Card, Suit, Rank
from foxzero.common import FoxZeroResNet

# UI Constants
SUIT_ICONS = {
    Suit.SPADE: "♠️",
    Suit.HEART: "❤️",
    Suit.DIAMOND: "♦️",
    Suit.CLUB: "♣️"
}

SUIT_NAMES_CN = {
    Suit.SPADE: "黑桃",
    Suit.HEART: "紅心",
    Suit.DIAMOND: "方塊",
    Suit.CLUB: "梅花"
}

def make_predict_batch_fn(model):
    def predict_batch_fn(obs_batch):
        inp = torch.from_numpy(obs_batch).to(next(model.parameters()).device)
        with torch.no_grad():
            l, v, b = model(inp)
            val = v.cpu().numpy().flatten()
            p_dist = torch.softmax(l, dim=1).cpu().numpy()
            b_probs = torch.sigmoid(b).cpu().numpy()
        return (p_dist, val, b_probs)
    return predict_batch_fn

def py_game_to_cpp_engine(py_game):
    import sevens_core
    engine = sevens_core.SevensEngine()
    engine.setupNewGame()
    
    engine.setFirstMovePerformed(py_game.first_move_performed)
    engine.setCurrentPlayer(py_game.current_player_number)
    engine.setDealer(py_game.dealer_number)
    engine.setTurnCount(py_game.turn_count)
    
    for i, py_hand in enumerate(py_game.hands):
        cards = [sevens_core.Card(c.suit, c.rank) for c in py_hand.cards]
        engine.setHand(i + 1, cards)
        
    for i, py_covered in enumerate(py_game.covered_cards):
        cards = [sevens_core.Card(c.suit, c.rank) for c in py_covered]
        engine.setCoveredCards(i + 1, cards)
        
    for suit in range(1, 5):
        ps = py_game.played_cards[suit - 1]
        low = ps.lowest_card.rank if ps.lowest_card else 0
        high = ps.highest_card.rank if ps.highest_card else 0
        engine.setPlayedCardRange(suit, low, high)
        
    return engine

def run_mcts_inference(game: SevensGame, model: FoxZeroResNet, simulations: int, c_puct=1.0, god_mode=True):
    import sevens_core
    
    current_player = game.current_player_number
    valid_moves = game.get_all_valid_moves(current_player)
    
    if len(valid_moves) == 0:
        return None
    if len(valid_moves) == 1:
        return valid_moves[0]
        
    predict_batch_fn = make_predict_batch_fn(model)
    engine = py_game_to_cpp_engine(game)
    
    best_card_idx = sevens_core.run_mcts_cpp(
        engine,
        predict_batch_fn,
        simulations,
        c_puct,
        god_mode,
        8 # num_threads
    )
    
    if best_card_idx == -1: return None
    
    s = best_card_idx // 13 + 1
    r = best_card_idx % 13 + 1
    return Card(s, r)


class Agent:
    def select_move(self, game: SevensGame, player_num: int) -> Card:
        pass

class RandomAgent(Agent):
    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        return np.random.choice(valid_moves)

class FoxZeroAgent(Agent):
    def __init__(self, model_path: str, simulations=400, c_puct=1.0, god_mode=False):
        self.device = torch.device("cpu")
        self.model = FoxZeroResNet()
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"已從 {model_path} 載入權重。")
            except Exception as e:
                print(f"警告：無法從 {model_path} 載入權重：{e}")
        else:
            print("使用隨機權重進行測試。")
        self.model.eval()
        self.model = torch.jit.script(self.model)
        self.simulations = simulations
        self.c_puct = c_puct
        self.god_mode = god_mode

    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        
        import time
        start_time = time.time()
        
        best_move = run_mcts_inference(
            game, 
            self.model, 
            simulations=self.simulations, 
            c_puct=self.c_puct, 
            god_mode=self.god_mode
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ MCTS 耗時: {elapsed:.2f} 秒 ({self.simulations} 次模擬)")
        return best_move

class HumanAgent(Agent):
    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            print(">>> 您沒有可以出的牌，必須 Pass。")
            return None
        
        # Display hand
        hand = sorted(game.hands[player_num - 1].cards, key=lambda c: (c.suit, c.rank))
        print(f"\n>>> 您的手牌 ({len(hand)} 張):")
        for i, card in enumerate(hand):
            icon = SUIT_ICONS.get(card.suit, str(card.suit))
            print(f"[{i}] {icon}{card.rank}", end="  ")
            if (i+1) % 7 == 0: print()
        print()
        
        # Display valid options
        print(">>> 可選動作:")
        options = {}
        for i, card in enumerate(valid_moves):
            icon = SUIT_ICONS.get(card.suit, str(card.suit))
            options[i] = card
            
            if game.is_valid_move(card):
                action_str = "出牌"
                suffix = ""
            else:
                action_str = "蓋牌"
                suffix = f" (扣分: {card.rank})"
                
            print(f"  ({i}) {action_str}: {icon}{card.rank}{suffix}")
        
        while True:
            try:
                choice = input(f">>> 請輸入編號 (0-{len(valid_moves)-1}): ")
                idx = int(choice)
                if idx in options:
                    return options[idx]
            except ValueError:
                pass
            except IndexError:
                pass
            print("無效輸入，請重新輸入。")

def format_card_cn(card: Card):
    if card is None: return "無"
    icon = SUIT_ICONS.get(card.suit, "?")
    return f"{icon}{card.rank}"

def print_board(game: SevensGame):
    print("\n" + "🏮" + " " + "—"*25 + " " + "🏮")
    print(f"  回合: {game.turn_count} | 輪到玩家 {game.current_player_number}")
    print("  " + "—"*27)
    print("  當前牌桌狀態:")
    # Suits order: Diamond, Club, Heart, Spade
    for suit in [Suit.SPADE, Suit.HEART, Suit.DIAMOND, Suit.CLUB]:
        ps = game.played_cards[suit - 1]
        name = SUIT_NAMES_CN.get(suit, "未知")
        icon = SUIT_ICONS.get(suit, "")
        
        if ps.lowest_card is None:
            print(f"  {icon} {name:2}: (空)")
        else:
            cards_str = [str(r) for r in range(ps.lowest_card.rank, ps.highest_card.rank + 1)]
            print(f"  {icon} {name:2}: {'-'.join(cards_str)}")
    print("🏮" + " " + "—"*25 + " " + "🏮")

def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="手動控制玩家 1")
    parser.add_argument("--sims", type=int, default=400, help="AI 思考次數 (MCTS Simulations)")
    parser.add_argument("--cpuct", type=float, default=1.0, help="MCTS 探索常數")
    parser.add_argument("--god", action="store_true", help="開啟上帝視角 (作弊模式)")
    args = parser.parse_args()
    
    # Handle conflicts
    god_mode = args.god

    model_path = "models/foxzero_weights.pth"
    if not os.path.exists(model_path):
        model_path = "models/foxzero_model.pth"
    
    # Agents setup
    if args.human:
        print("模式：玩家 1 (您) vs 3 位 FoxZero AI")
        p1_agent = HumanAgent()
    else:
        print("模式：FoxZero AI vs 3 位 FoxZero AI")
        p1_agent = FoxZeroAgent(model_path, simulations=args.sims, c_puct=args.cpuct, god_mode=god_mode)
        
    # All opponents are now FoxZero (Strong) to test capability
    # Or keep Random? "FoxZero AI vs 3 位隨機 AI" in original.
    # User said: "Transform it into a demon".
    # Usually we want THE AI (Player 1 or AI) to be the demon.
    # Let's keep P1 as Hero (Human/AI) and others as Random by default?
    # Or upgrade opponents?
    # Original: p1 vs Randoms.
    # Let's keep opponents Random for now to demonstrate P1 dominance, or maybe P2/3/4 use weak MCTS?
    # Let's stick to Random for opponents so valid comparison can be made, or upgrade if user wants.
    # The prompt implies: "Modify play.py so *AI* can play..."
    # If I am Human (P1), I want to play AGAINST the "Demon".
    # So P2, P3, P4 should be the FoxZeroAgents.
    
    # Wait, the prompt says "make IT [The AI] a demon".
    # If I run --human, I am P1.
    # So P2, P3, P4 MUST be FoxZeroAgents to challenge me.
    # If I run AI vs AI, then P1 should be FoxZero.
    
    # Revised Logic:
    # If human: P1=Human, P2-4=FoxZero(Sims=args.sims).
    # If AI: P1=FoxZero, P2-4=Random (to show P1 dominance) or FoxZero (Clash of Gods)?
    # Let's make P2-4 FoxZero as well in Human mode?
    # Actually, calculating 3 AIs * 400 sims takes time on CPU.
    # Let's make P2, P3, P4 FoxZeroAgent if Human is playing, so he feels the pain.
    # But for "AI vs AI" (default), usually showcases one strong agent vs weak.
    
    ai_agent = FoxZeroAgent(model_path, simulations=args.sims, c_puct=args.cpuct, god_mode=god_mode)
    random_agent = RandomAgent()
    
    agents = {}
    if args.human:
        agents[1] = HumanAgent() 
        # Making opponents strong
        agents[2] = ai_agent
        agents[3] = ai_agent
        agents[4] = ai_agent
        print(f"對手等級: FoxZero (Sims={args.sims}, GodMode={god_mode})")
    else:
        agents[1] = ai_agent
        agents[2] = random_agent
        agents[3] = random_agent
        agents[4] = random_agent
        print(f"主角等級: FoxZero (Sims={args.sims}, GodMode={god_mode})")
        print("對手等級: Random (沙包)")
    
    game = SevensGame()
    print("\n遊戲開始！")
    print(f"莊家 (持黑桃 7): 玩家 {game.dealer_number}")
    
    while not game.is_game_over():
        current = game.current_player_number
        print_board(game)
        
        agent = agents[current]
        
        if isinstance(agent, FoxZeroAgent):
            mode_str = "思考中 (GodMode)" if god_mode else "思考中 (MCTS + Logic)"
            print(f"AI (玩家 {current}) {mode_str} (Sims={args.sims})...")
            
        card = agent.select_move(game, current)
        
        if card is None:
            print(f"💥 玩家 {current} 通過 (Pass)")
            game.record_pass(current)
        else:
            if game.is_valid_move(card):
                print(f"✅ 玩家 {current} 出牌: {format_card_cn(card)}")
            else:
                score = card.rank
                print(f"⚠️ 玩家 {current} 蓋牌: {format_card_cn(card)} (扣分: {score})")
            
            game.make_move(card)
            
        game.next_player()
        if not args.human:
            time.sleep(0.5)
        
        if game.turn_count > 300:
            print("防止無限迴圈，強制結束。")
            break
            
    print("\n" + "🏁" * 15)
    print("遊戲結束！")
    rewards = game.calculate_final_rewards()
    
    max_reward = -float('inf')
    best_player = -1
    for i in range(4):
        if rewards[i] > max_reward:
            max_reward = rewards[i]
            best_player = i + 1
            
    for i in range(1, 5):
        hand_score = sum(c.rank for c in game.hands[i-1].cards)
        cover_score = sum(c.rank for c in game.covered_cards[i-1])
        penalty = hand_score + cover_score
        
        is_winner = (i == best_player)
        winner_mark = " 🏆 (贏家)" if is_winner else ""
        
        ender_mark = " 🔚 (結束局)" if game.has_player_won(i) else ""
        
        print(f"玩家 {i}: 獎勵={rewards[i-1]:.4f}, 總點數={penalty} (手牌 {hand_score} + 蓋牌 {cover_score}){winner_mark}{ender_mark}")
    print("🏁" * 15)

if __name__ == "__main__":
    main()

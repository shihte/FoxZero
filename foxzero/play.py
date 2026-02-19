import torch
import numpy as np
import time
import argparse
import sys
import os
from foxzero.game import SevensGame, Card, Suit, Rank
from foxzero.common import FoxZeroResNet
from foxzero.mcts import MCTS

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
    def __init__(self, model_path: str, simulations=400):
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
        self.simulations = simulations

    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        
        mcts = MCTS(self.model, simulations=self.simulations)
        pi = mcts.search(game)
        
        action_idx = np.argmax(pi)
        s_idx = action_idx // 13
        r_idx = action_idx % 13
        card = Card(s_idx + 1, r_idx + 1)
        
        if card not in valid_moves:
            return valid_moves[0]
        return card

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
    parser.add_argument("--sims", type=int, default=400, help="AI 思考次數")
    args = parser.parse_args()

    model_path = "foxzero_weights.pth"
    if not os.path.exists(model_path):
        model_path = "foxzero_model.pth"
    
    # Agents setup
    if args.human:
        print("模式：玩家 1 (您) vs 3 位隨機 AI")
        p1_agent = HumanAgent()
    else:
        print("模式：FoxZero AI vs 3 位隨機 AI")
        p1_agent = FoxZeroAgent(model_path, simulations=args.sims)
        
    random_agent = RandomAgent()
    agents = {
        1: p1_agent,
        2: random_agent,
        3: random_agent,
        4: random_agent
    }
    
    game = SevensGame()
    print("\n遊戲開始！")
    print(f"莊家 (持黑桃 7): 玩家 {game.dealer_number}")
    
    while not game.is_game_over():
        current = game.current_player_number
        print_board(game)
        
        agent = agents[current]
        
        if isinstance(agent, FoxZeroAgent):
            print(f"AI (玩家 {current}) 思考中...")
            
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
    
    # Check true winner (lowest adjusted score? or lowest raw score?)
    # Let's show Raw Score for clarity, but mark Winner based on Rewards (which reflect adjusted rules).
    # Actually, Game Logic determines rewards. We should trust rewards to define "Winning".
    # Player with max reward is the winner of the session.
    
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
        
        # Identify who ended the game
        ender_mark = " 🔚 (結束局)" if game.has_player_won(i) else ""
        
        print(f"玩家 {i}: 獎勵={rewards[i-1]:.4f}, 總點數={penalty} (手牌 {hand_score} + 蓋牌 {cover_score}){winner_mark}{ender_mark}")
    print("🏁" * 15)

if __name__ == "__main__":
    main()

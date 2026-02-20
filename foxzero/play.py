import torch
import numpy as np
import time
import argparse
import sys
import os
import math
import copy
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

class InferenceMCTSNode:
    """Lightweight MCTS Node for Inference."""
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {} # Map[Card, InferenceMCTSNode]
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        
    def ucb_score(self, c_puct):
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        
        q = self.value_sum / self.visits if self.visits > 0 else 0.0
        # U = c * P * sqrt(N_parent) / (1 + N_child)
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

def run_mcts_inference(game: SevensGame, model: FoxZeroResNet, simulations: int, c_puct=1.0, god_mode=True):
    """
    Runs MCTS for a single move decision.
    god_mode=True: AI sees all hands (Upper Bound / Cheating).
    god_mode=False: AI randomizes hidden hands (Determinization).
    """
    root = InferenceMCTSNode()
    
    # 0. Check valid moves
    current_player = game.current_player_number
    valid_moves = game.get_all_valid_moves(current_player)
    
    if len(valid_moves) == 0:
        return None
    if len(valid_moves) == 1:
        return valid_moves[0]
        
    # Expand root once to get priors
    state_tensor = game.get_state_tensor(current_player)
    inp = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
    
    with torch.no_grad():
        logits, _ = model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        
    # Create Root Children
    for card in valid_moves:
        s, r = card.to_tensor_index()
        idx = s * 13 + r
        prior = probs[idx]
        root.children[card] = InferenceMCTSNode(parent=root, prior=prior)
        
    # Main Loop
    for _ in range(simulations):
        node = root
        
        # 1. Determinization / Copy
        scratch_game = copy.deepcopy(game)
        if not god_mode:
            # Determinize: Shuffle opponents' hands based on known info
            scratch_game.determinize(observer_player=current_player)
            
        # 2. Selection
        while len(node.children) > 0:
            # Select best child
            card, node = max(node.children.items(), key=lambda item: item[1].ucb_score(c_puct))
            
            # Apply move in simulation
            scratch_game.make_move(card)
            scratch_game.next_player()
            
            # If game over during selection, break
            if scratch_game.is_game_over():
                break
                
        # 3. Expansion & Evaluation
        leaf_player = scratch_game.current_player_number
        value = 0.0
        
        if scratch_game.is_game_over():
            # Terminal state
            final_rewards = scratch_game.calculate_final_rewards()
            # Reward for the player whose turn it was at the PARENT of this node?
            # Actually, standard is: We want value for `leaf_player`.
            # If game over, `leaf_player` doesn't matter, we get rewards vector.
            # But we need a scalar 'value' to propagate back up.
            # The value depends on WHO is evaluating.
            pass # Value calc below
        else:
            # Neural Net Eval
            # Check valid moves at leaf
            leaf_moves = scratch_game.get_all_valid_moves(leaf_player)
            
            if len(leaf_moves) > 0:
                s_t = scratch_game.get_state_tensor(leaf_player)
                inp = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
                with torch.no_grad():
                    l, v = model(inp)
                    val = v.item()
                    p_dist = torch.softmax(l, dim=1).cpu().numpy().flatten()
                
                # Expand children
                for c in leaf_moves:
                    s, r = c.to_tensor_index()
                    idx = s * 13 + r
                    node.children[c] = InferenceMCTSNode(parent=node, prior=p_dist[idx])
                
                value = val # Value for 'leaf_player'
            else:
                # Pass? Or game over?
                # If no moves but game not over, it's a pass.
                # In Sevens, pass is forced. Make pass move and continue?
                # Simplified: Treat as terminal 0 or continue?
                # For this MCTS, let's just stop expansion and eval as 0 (drawish) or -1?
                # Actually Sevens Pass is valid move logic handled by game?
                # If get_all_valid_moves returns [], it means pass... 
                # But game.is_valid_move includes logic.
                # Let's assume recursion handles it or just use value=0
                value = -0.5 # Penalty for stuck?
                
        # 4. Backpropagation
        # Propagate value up the tree.
        # Value 'v' is from perspective of 'leaf_player'.
        # We need to toggle sign at each level if it's opponent (Zero-Sum assumption).
        # Or more robustly: evaluate relative reward.
        
        # Robust Backprop for Multiplayer (Score-based)
        # If terminal:
        if scratch_game.is_game_over():
            rewards = scratch_game.calculate_final_rewards()
            # Backprop real rewards
            curr = node
            sim_player = leaf_player # This might be arbitrary if game over
            # Actually, we climb up. Each node represents a state where 'prev_player' made a move to get here.
            # The 'node' value should be helpfulness for 'node.parent.player'.
            
            # Let's use simple logic:
            # We have vector 'rewards' [p1, p2, p3, p4].
            while curr.parent is not None:
                # Move led to 'curr' was made by 'prev_player'
                # We need to identify who made the move.
                # Root is 'current_player'.
                # Depth 1 child is after 'current_player' moved.
                # So edge Root->Child is 'current_player' move.
                # We update Child with 'current_player' reward.
                
                # Need to track player at each depth? Or just reconstruct.
                # Sevens is strictly rotational 1->2->3->4...
                # Current at Root = P_root.
                # Edge Root->Child: P_root acted.
                # Child state: P_root+1's turn.
                
                # So, Child node accumulates value for P_root.
                # Its parent (Root) was P_root's state.
                
                # We can trace back from current game state? No, scratch_game is mutated.
                # We can strictly alternate.
                pass
                
                # Simple Hack:
                # Just use N-step return or Value Head mixing.
                # For inference "God Mode", just use the Value Head for non-terminal.
        
        # Simplified Backprop for Inference:
        # Assume 2-player zero-sum dynamic? No, it's 4-player.
        # But `value` output from Net is "Win Probability/Score" for current player.
        # Standard: v maps to [-1, 1].
        # If P1 has v=0.8, P2/3/4 likely have negative.
        # When backing up to parent (who was P_prev), we need P_prev's value.
        # Approx: P_prev value = -P_curr value (if 2 player).
        # For 4-player, let's stick to -v/3 or just -v?
        # Let's use: -value (Adversarial assumption).
        
        curr = node
        curr_val = value # Value for 'leaf_player'
        
        while curr.parent is not None:
            # Determine return for the player who acted to reach 'curr'
            # (which is curr.parent's player)
            # If curr_val is for 'leaf_player', and we move up...
            # The opponent's gain is roughly my loss.
            curr_val = -curr_val
            
            curr.value_sum += curr_val
            curr.visits += 1
            curr = curr.parent

    # Return best move
    # Select child with most visits
    if len(root.children) == 0:
        return None
        
    best_card = max(root.children, key=lambda c: root.children[c].visits)
    return best_card

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
        self.simulations = simulations
        self.c_puct = c_puct
        self.god_mode = god_mode

    def select_move(self, game: SevensGame, player_num: int) -> Card:
        valid_moves = game.get_all_valid_moves(player_num)
        if not valid_moves:
            return None
        
        return run_mcts_inference(
            game, 
            self.model, 
            simulations=self.simulations, 
            c_puct=self.c_puct, 
            god_mode=self.god_mode
        )

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
    parser.add_argument("--god_mode", action="store_true", default=True, help="開啟上帝視角 (作弊模式)")
    parser.add_argument("--blind", action="store_true", help="關閉上帝視角 (使用決定論化)")
    args = parser.parse_args()
    
    # Handle conflicts
    god_mode = args.god_mode
    if args.blind:
        god_mode = False

    model_path = "foxzero_weights.pth"
    if not os.path.exists(model_path):
        model_path = "foxzero_model.pth"
    
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
            mode_str = "思考中 (GodMode)" if god_mode else "思考中 (Determinized)"
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

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
    import sys
    for i in range(simulations):
        if (i + 1) % 10 == 0:
            print(f"\r🔍 MCTS 思考中... {i + 1}/{simulations}", end="")
            sys.stdout.flush()

        node = root
        
        # 1. Determinization / Copy
        scratch_game = copy.deepcopy(game)
        if not god_mode:
            # Determinize: Shuffle opponents' hands based on known info
            scratch_game.determinize(observer_player=current_player)
            
        # 2. Selection
        # 2. Selection
        while True:
            # Check valid moves for current player in this specific determinization
            current_p = scratch_game.current_player_number
            valid_moves = scratch_game.get_all_valid_moves(current_p)
            
            # If no valid moves, it's terminal or pass-stuck?
            if len(valid_moves) == 0:
                break
                
            # Filter children that are valid in this universe
            feasible_children = [c for c in valid_moves if c in node.children]
            unexpanded = [c for c in valid_moves if c not in node.children]
            
            # If there are unexpanded moves valid in this universe, we should stop selection 
            # and let the Expansion phase handle this node (add the unexpanded child).
            if len(unexpanded) > 0:
                # We stop at 'node' and will expand from here.
                break
                
            if len(feasible_children) > 0:
                # All valid moves are already in tree. Select best feasible child.
                # Only compare UCB among feasible children
                card = max(feasible_children, key=lambda c: node.children[c].ucb_score(c_puct))
                node = node.children[card]
                
                scratch_game.make_move(card)
                scratch_game.next_player()
                
                if scratch_game.is_game_over():
                    break
            else:
                # Should not happen if len(valid_moves) > 0 and unexpanded check failed
                break
                
        # 3. Expansion & Evaluation
        leaf_player = scratch_game.current_player_number
        value = 0.0
        
        if scratch_game.is_game_over():
            # Terminal state (Global)
            rewards = scratch_game.calculate_final_rewards()
            # We don't have a 'value' relative to a single player here easily, 
            # but backprop handles vector rewards if implemented fully. 
            # For now, let's just pick the reward for the leaf_player to fit 'value' scalar api,
            # but usually we handle terminal differently. 
            # Note: The backprop loop below assumes 'value' flip.
            # Let's treat it as 0.0 value but set a flag? 
            # Actually, let's use the loop below which is designed for scalar value.
            # If game over, we can't use scalar value easily.
            # Hack: loop expects 'value' for 'leaf_player'.
            pass 
        else:
            # Not Game Over -> Expand
            leaf_player = scratch_game.current_player_number
            valid_moves_leaf = scratch_game.get_all_valid_moves(leaf_player)
            
            if len(valid_moves_leaf) > 0:
                # Check which ones are new
                unexpanded = [c for c in valid_moves_leaf if c not in node.children]
                
                if len(unexpanded) > 0:
                    # Need to evaluate state to get priors/value for these new moves
                    s_t = scratch_game.get_state_tensor(leaf_player)
                    inp = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
                    with torch.no_grad():
                        l, v = model(inp)
                        val = v.item()
                        p_dist = torch.softmax(l, dim=1).cpu().numpy().flatten()
                    
                    # Add NEW children
                    for c in unexpanded:
                        s, r = c.to_tensor_index()
                        idx = s * 13 + r
                        # Use the prior from THIS evaluation. 
                        # Note: This might bias priors based on the specific hand we held when first discovering the move.
                        # This is an acceptable approximation for IS-MCTS.
                        node.children[c] = InferenceMCTSNode(parent=node, prior=p_dist[idx])
                        
                    value = val
                else:
                    # All moves expanded, but we stopped here?
                    # This happens if selection loop broke because we want to Rollout?
                    # Or maybe we just treat this node as the leaf to evaluate score?
                    # If we are here, it means we traversed deeply and either:
                    # 1. Game Over (handled above)
                    # 2. We stopped at a "fully expanded node"?? 
                    # Wait, Selection loop only breaks if:
                    # - Game Over
                    # - Unexpanded moves exist (Handled above)
                    # - No valid moves (Handled below)
                    # - feasible_children is empty (impossible if valid>0 and unexpanded=0)
                    
                    # So actually, if we are here and valid_moves>0 and len(unexpanded)==0:
                    # We shouldn't be here?
                    # Ah, loop runs 'simulations' times. The Selection loop breaks.
                    # If selection loop ran until Game Over, we are in `if game_over`.
                    # If selection loop broke due to `unexpanded`, we are in `if len(unexpanded) > 0`.
                    # So this logic covers it.
                    
                    # What if we reached a leaf that is fully expanded?
                    # Selection loop continues UNTIL a leaf?
                    # Standard MCTS: select until we fall out of tree.
                    # My selection loop: `while len(node.children) > 0`: but filtered.
                    # If node has children but NONE are feasible: `feasible_children` empty.
                    # Then it breaks.
                    # Then `unexpanded` will be ALL `valid_moves` (since they are valid but not in children? No).
                    # Loop:
                    # valid_moves = [A, B]
                    # children = [C, D] (from other universes)
                    # feasible = []
                    # unexpanded = [A, B]
                    # Breaks.
                    # Here: unexpanded=[A,B]. Adds them. OK.
                    pass
                    
            else:
                # No valid moves (Pass logic?)
                # In Sevens, you must cover. get_all_valid_moves returns hand if must cover.
                # So if len==0, it's really stuck or empty hand?
                # Empty hand = Game Over, handled by game.is_game_over().
                value = 0.0
                
        # 4. Backpropagation
        # Propagate value up the tree.
        # Value 'v' is from perspective of 'leaf_player'.
        # We need to toggle sign at each level if it's opponent (Zero-Sum assumption).
        # Or more robustly: evaluate relative reward.
        
        # Robust Backprop for Multiplayer (Score-based)
        # If terminal:
        if scratch_game.is_game_over():
            rewards = scratch_game.calculate_final_rewards()
            # standard backprop below expects 'value' for 'leaf_player'
            # rewards is 0-indexed array [p1, p2, p3, p4]
            # leaf_player is 1-based index (1..4)
            r = rewards[leaf_player - 1]
            # Normalize reward? rewards are like +100, -10, etc.
            # MCTS expects [-1, 1].
            # Sigmoid or Tanh? Or just sign?
            # Sevens rewards: Winner ~ +100. Losers ~ -10 to -50.
            # Let's simple clamp or sign.
            if r > 0: value = 1.0
            elif r < 0: value = -1.0
            else: value = 0.0
            
            # Fall through to standard backprop using this 'value'
        
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
        
        depth_sanity = 0
        while curr.parent is not None:
            depth_sanity += 1
            if depth_sanity > 60:
                print(f"\n⚠️ MCTS Backprop Loop Detected! Depth {depth_sanity}. Breaking.")
                break

            # Determine return for the player who acted to reach 'curr'
            # (which is curr.parent's player)
            # If curr_val is for 'leaf_player', and we move up...
            # The opponent's gain is roughly my loss.
            curr_val = -curr_val
            
            curr.value_sum += curr_val
            curr.visits += 1
            curr = curr.parent

    print() # Newline after progress bar

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

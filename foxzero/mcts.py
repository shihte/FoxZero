import math
import numpy as np
import copy
from typing import Dict, List, Tuple
from foxzero.game import SevensGame, Card
from foxzero.common import FoxZeroResNet

class MCTSNode:
    def __init__(self, parent=None, prior_prob=0.0):
        self.parent = parent
        self.children: Dict[Card, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.0):
        if self.parent is None:
            return 0.0
        
        # Standard PUCT formula
        # Q(s, a) + U(s, a)
        # U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        u = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + u

class MCTS:
    def __init__(self, net: FoxZeroResNet, simulations=800, c_puct=1.0):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, game: SevensGame) -> List[float]:
        """
        Runs MCTS simulations and returns action probabilities (pi).
        Returns list of 52 probabilities corresponding to 52 cards (sorted by tensor index).
        """
        root = MCTSNode()
        
        # Optimization: Expand root once to check valid moves
        # If only 1 valid move, returns one-hot directly?
        # The FoxZero design says: "Single Move: Skip MCTS search... but must store (s, a)".
        # To support training, we should return the probability 1.0 for that move.
        current_player = game.current_player_number
        valid_moves = game.get_all_valid_moves(current_player)
        
        if len(valid_moves) == 0:
            # Should not happen in Sevens unless game over, but check
             return np.zeros(52)
             
        if len(valid_moves) == 1:
            # Single move optimization
            pi = np.zeros(52)
            c = valid_moves[0]
            s_idx, r_idx = c.to_tensor_index()
            idx = s_idx * 13 + r_idx
            pi[idx] = 1.0
            return pi

        # Run simulations
        for _ in range(self.simulations):
            node = root
            
            # Determinization:
            # Create a view of the game state consistent with the current player's knowledge.
            # If current_player is the root player (who is thinking), we know their hand.
            # For others, we shuffle the UNKNOWN cards.
            # However, MCTS tree nodes represent specific moves.
            # Usually, IS-MCTS (Information Set MCTS) works by sampling a world state at the START of each simulation.
            
            scratch_game = copy.deepcopy(game) 
            
            # TODO: Implement determinize() in SevensGame to shuffle unplayed cards among opponents?
            # Ideally:
            scratch_game.determinize(observer_player=current_player)
            
            # 1. Selection
            
            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                # Select best child
                node = max(node.children.values(), key=lambda n: n.ucb_score(self.c_puct))
                
                # Find the action (card) that led to this node
                # We need to store action in node or iterate children?
                # Let's iterate parent's children to find the card
                action_card = None
                for card, child in node.parent.children.items():
                    if child is node:
                        action_card = card
                        break
                
                scratch_game.make_move(action_card)
                scratch_game.next_player()

            # 2. Expansion & Evaluation
            current_player_at_leaf = scratch_game.current_player_number
            
            if scratch_game.is_game_over():
                # Terminal state
                rewards = scratch_game.calculate_final_rewards() # [p1_r, p2_r, p3_r, p4_r]
                self._backpropagate(node, rewards, current_player_at_leaf, game.num_players)
            else:
                # Expand
                state_tensor = scratch_game.get_state_tensor(current_player_at_leaf)
                
                # Predict
                policy_logits, value = self.net.predict(state_tensor) 
                # policy_logits is [52], value is float
                
                # Filter valid moves
                valid_moves_leaf = scratch_game.get_all_valid_moves(current_player_at_leaf)
                
                policy_map = {} # card -> prob
                sum_prob = 0.0
                
                for card in valid_moves_leaf:
                    s_idx, r_idx = card.to_tensor_index()
                    idx = s_idx * 13 + r_idx
                    prob = policy_logits[idx] 
                    # If logits are raw, use exp? 
                    # In model.py we did Softmax at the end of forward! 
                    # So prediction is already probability.
                    policy_map[card] = prob
                    sum_prob += prob
                
                # Renormalize
                if sum_prob > 0:
                    for card in policy_map:
                        policy_map[card] /= sum_prob
                else:
                    # Uniform if something went wrong (e.g. all masked out?)
                    for card in policy_map:
                        policy_map[card] = 1.0 / len(valid_moves_leaf)
                        
                # Create children
                node.is_expanded = True
                for card, prob in policy_map.items():
                    child = MCTSNode(parent=node, prior_prob=prob)
                    node.children[card] = child
                
                # Setup reward vector for backprop
                # Value head predicts reward for CURRENT player.
                # We need to construct a reward vector where other players get ... what?
                # FoxZero is zero-sum.
                # If value is V for current player p, assume zero-sum distribution for others?
                # Or simply backpropagate `v` and only update nodes belonging to `p`?
                # NO. In MCTS, every node needs a value update.
                # If node represents P1's turn, we update Q with Reward for P1.
                # If we have V (for P_leaf), we can assume it's the reward for P_leaf.
                # What about other players? 
                # For 4-player games, maybe we assume simplified zero-sum? 
                # Or just update Q with V if node.player == leaf.player, and -V/3 otherwise?
                # Let's use specific logic:
                # If we assume 2-player zero sum, r_opp = -r_me.
                # For 4-player, "Normalized Rewards" sum to 0. 
                # So if P1 gets V, others get -V/3 roughly?
                # Let's approximate: 
                # rewards = [0]*4
                # rewards[current_player_at_leaf - 1] = value
                # for others: rewards[i] = -value / 3.0
                # This ensures zero sum.
                
                rewards = [0.0] * 4
                p_idx = current_player_at_leaf - 1
                rewards[p_idx] = value
                for i in range(4):
                    if i != p_idx:
                        rewards[i] = -value / 3.0
                        
                self._backpropagate(node, rewards, current_player_at_leaf, game.num_players)

        # 3. Get Policy from Root
        pi = np.zeros(52)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        # Temperature tau (handled by caller? or inside?)
        # "前 30 步: tau=1.0... 30步後: tau->0"
        # Since this method returns prob distribution, let's return count-based distribution (tau=1) 
        # and let caller handle greedy/sampling.
        
        if total_visits > 0:
            for card, child in root.children.items():
                s_idx, r_idx = card.to_tensor_index()
                idx = s_idx * 13 + r_idx
                pi[idx] = child.visit_count / total_visits
                
        return pi

    def _backpropagate(self, node: MCTSNode, rewards: List[float], leaf_player_num: int, num_players: int):
        # Traversing up via parent...
        # Wait, I need to know WHOSE turn it was at 'node'.
        # 'node' corresponds to the state BEFORE the action was chosen?
        # No, 'node' is the state.
        # But `node` object doesn't store state or player. 
        # We need to know the player at each node in the path.
        # We can reconstruct it or simply propagate up.
        # 
        # Correct MCTS path:
        # Root (Player P) ->(Action A)-> Child (Player Q) -> ...
        # If we are at `node` (Child), its value `Q` should represent value for Player P (parent's player)?
        # No, typically Q(s, a) estimates the value of taking action `a` from `s`.
        # So Q is value for Player At State `s` (Parent).
        # So when updating `child`, we update it with Reward for `child.parent`'s player.
        # 
        # In multiplayer, usually we store Q as vector [v1, v2, v3, v4]?
        # Or simpler:
        # Each node stores "Value for the player who executes move FROM this node".
        # Root node: player P1. Q(Root->Child) is value for P1.
        # So we update Child W/N with P1's reward.
        
        # To do this, we need to know P1 at Root.
        # The players rotate: P1, P2, P3, P4...
        # We can calculate back which player acted.
        
        curr = node
        # The player at 'leaf' was 'leaf_player_num' (before expanding).
        # So 'node' is a state where it is 'leaf_player_num' turn.
        # But 'node' has parents.
        # Let's re-verify:
        # Root (Turn P1). Selects Child (Turn P2). 
        # Action at Root was by P1.
        # Q(Child) stores value for P1.
        
        # Traverse UP:
        # Node (Leaf, P_leaf). Parent (P_leaf-1).
        # Parent->Node edge represents action by P_leaf-1.
        # So update Node with Reward[P_leaf-1].
        
        # Logic:
        # current player at 'leaf node' is `p`.
        # We go up.
        # `p` = `p` - 1 (Previous player).
        # Update node with `rewards[p]`.
        
        player = leaf_player_num
        
        while curr.parent is not None:
            # Move up to parent
            # The edge parent->curr was an action by 'prev_player'
            # Who is 'prev_player'? The player at 'parent'.
            prev_player_idx = (player - 2) % num_players # player 1-4 -> idx 0-3...
            # e.g. player=1 (P1). Prev is 4. (1-2)%4 = -1%4 = 3 (P4). Correct.
            
            player_idx_for_update = prev_player_idx
            
            reward = rewards[player_idx_for_update]
            
            curr.value_sum += reward
            curr.visit_count += 1
            
            curr = curr.parent
            player = (player - 2) % num_players + 1 # Update player number for next step up

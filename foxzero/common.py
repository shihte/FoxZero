import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from foxzero.game import SevensGame, Card

# -------------------------
# Neural Network Definition
# -------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class FoxZeroResNet(nn.Module):
    def __init__(self, num_res_blocks=5, num_channels=256):
        super().__init__()
        
        # Backbone
        # Input: [Batch, 11, 4, 13]
        self.conv_input = nn.Conv2d(11, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 4 * 13, 52) # 52 cards
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 4 * 13, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Belief Head (Predicts 3 opponents * 4 suits * 13 ranks)
        self.belief_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.belief_bn = nn.BatchNorm2d(2)
        self.belief_fc = nn.Linear(2 * 4 * 13, 3 * 4 * 13)

    def forward(self, x):
        # x shape: [Batch, 11, 4, 13]
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        # Belief Head
        b = F.relu(self.belief_bn(self.belief_conv(x)))
        b = b.view(b.size(0), -1)
        b = self.belief_fc(b)
        b = torch.sigmoid(b)
        b = b.view(-1, 3, 4, 13)
        
        return p, v, b

    def predict(self, state_tensor):
        """
        Inference wrapper.
        state_tensor: np.array [11, 4, 13]
        Returns: policy (np.array [52]), value (float)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_tensor).unsqueeze(0) # Add batch dim
            p, v, b = self(x)
            return p.squeeze(0).numpy(), v.item(), b.squeeze(0).numpy()

# -------------------------
# Simulation Logic
# -------------------------

def run_mcts_game_simulation(game_cls, model, sims):
    """
    Runs a single game simulation using MCTS.
    Returns list of samples: (state, policy, final_reward_for_player)
    
    Args:
        game_cls: Class to instantiate game (e.g. SevensGame)
        model: FoxZeroResNet instance (CPU)
        sims: Number of MCTS simulations
    """
    # Import locally to avoid circular import (MCTS imports FoxZeroResNet)
    from foxzero.mcts import MCTS
    
    game = game_cls()
    mcts = MCTS(model, simulations=sims)
    
    game_samples = [] # (state, pi, player_idx, belief_target)
    step_count = 0
    
    while not game.is_game_over():
        current_player = game.current_player_number
        
        # MCTS Search
        pi = mcts.search(game)
        state_tensor = game.get_state_tensor(current_player)
        belief_target = np.array(game.get_belief_target(current_player), dtype=np.float32).reshape(3, 4, 13)
        
        game_samples.append((state_tensor, pi, current_player, belief_target))
        
        # Select Action
        valid_moves = game.get_all_valid_moves(current_player)
        
        if len(valid_moves) == 0:
            game.record_pass(current_player)
            game.next_player()
            continue
            
        # Sampling (Temperature=1 for first 30 moves, then Argmax)
        if step_count < 30:
            action_idx = np.random.choice(len(pi), p=pi)
        else:
            action_idx = np.argmax(pi)
        
        # Convert to Card
        s_idx = action_idx // 13
        r_idx = action_idx % 13
        card = Card(s_idx + 1, r_idx + 1)
        
        if card not in valid_moves:
            # Fallback (should rarely happen if MCTS is correct)
            card = valid_moves[0]
            
        game.make_move(card)
        game.next_player()
        step_count += 1
        
        if step_count > 300: # Loop prevention
            break
            
    # Calculate Rewards
    if game.is_game_over():
        final_rewards = game.calculate_final_rewards()
        
        # Build samples
        final_samples = []
        for s, pi, p_idx, b_targ in game_samples:
            z = final_rewards[p_idx - 1]
            final_samples.append((s, pi, b_targ, z))
        return final_samples
    else:
        return []

def run_simulation_fast(game_cls, model, temperature=1.0, dirichlet_alpha=None, top_k=None):
    """
    Runs a simulation using direct Model Policy Sampling (No MCTS search).
    Optimized for Colab CPU workers where MCTS is too slow.
    Uses 'sevens_core' C++ engine if available, else 'SevensGame' (Python).
    """
    # Try C++ Core
    cpp_engine = None
    try:
        import sevens_core
        # specific check if game_cls is meant to be Python or if we can swap
        # We prefer C++ if available and game_cls is standard SevensGame
        if game_cls.__name__ == 'SevensGame' or game_cls is None:
             cpp_engine = sevens_core.SevensEngine()
    except ImportError:
        pass
        
    model.eval()
    device = next(model.parameters()).device
    traj = [] # (state_tensor, policy_probs, player_id, belief_target)
    
    if cpp_engine:
        # --- C++ PATH ---
        game = cpp_engine
        while not game.isGameOver():
            
            p_idx = game.getCurrentPlayerNumber() # 1-4
            
            # 1. Observation
            obs_list = game.get_observation(p_idx) # Flat list
            # Reshape (11, 4, 13)
            # It returns flat 11*4*13.
            state_tensor = np.array(obs_list, dtype=np.float32).reshape(11, 4, 13)
            
            inp = torch.tensor(state_tensor).unsqueeze(0).to(device)
            
            # 2. Model
            with torch.no_grad():
                p_logits, _, _ = model(inp)
                
            # TEMPERATURE CONTROL
            # Decrease temp in endgame to reduce randomness
            current_temp = temperature
            if len(traj) >= 30:
                current_temp = 0.5 # Anneal
                
            # Apply Temp to Logits (before Softmax)
            p_logits = p_logits / current_temp
            
            p_probs = torch.softmax(p_logits, dim=1).cpu().numpy().flatten()
            
            # 3. Mask
            legal_mask = np.array(game.get_legal_moves(), dtype=np.float32) # Returns flat 52-float mask
            
            p_probs = p_probs * legal_mask
            if p_probs.sum() > 0:
                p_probs /= p_probs.sum()
            else:
                if legal_mask.sum() > 0:
                    p_probs = legal_mask / legal_mask.sum()
                else:
                    break
                    
            # DIRICHLET NOISE (Exploration)
            # Only add noise in early game (first 30 moves) and if alpha is set
            if dirichlet_alpha is not None and len(traj) < 30:
                noise = np.random.dirichlet([dirichlet_alpha] * 52)
                p_probs = 0.75 * p_probs + 0.25 * noise
                # Re-mask and Re-normalize to ensure legality
                p_probs = p_probs * legal_mask
                if p_probs.sum() > 0:
                    p_probs /= p_probs.sum()
                else:
                     p_probs = legal_mask / legal_mask.sum()

            # TOP-K FILTERING (Prune low-prob moves)
            if top_k is not None and top_k > 0:
                # Get indices of NOT top-k elements
                if top_k < len(p_probs):
                    # Sort indices by value descending
                    sorted_indices = np.argsort(p_probs)[::-1]
                    # Keep valid ones only? p_probs matches 52 cards.
                    # Just zero out everything after k-th element
                    # But wait, legal_mask might make some 0 already.
                    # We only care about active probabilities.
                    
                    # Simply zero out indices not in top_k
                    mask_indices = sorted_indices[top_k:]
                    p_probs[mask_indices] = 0.0
                    
                    # Re-normalize
                    if p_probs.sum() > 0:
                        p_probs /= p_probs.sum()
                    else:
                        # If top_k removed everything (unlikely if k>=1 and sum>0 initially)
                        # Maybe all probs were 0?
                         p_probs = legal_mask / legal_mask.sum()

            # 4. Sample
            action_idx = np.random.choice(52, p=p_probs)
            
            b_target = np.array(game.get_belief_target(p_idx), dtype=np.float32).reshape(3, 4, 13)
            traj.append((state_tensor, p_probs, p_idx, b_target))
            
            # 5. Step
            game.step(action_idx)
            
        # --- REWARD HARD FIX (Winner Takes All) ---
        # Calculate raw scores for all players
        scores = {}
        for p in range(1, 5):
            # Score = Hand Card Ranks + Covered Card Ranks
            # precise calculation from game state
            hand = game.getPlayersHand(p)
            s = 0
            for c in hand.getHand():
                s += c.getRank()
            
            # Add covered cards
            # We need to access covered cards from C++ engine? 
            # `sevens_python.cpp` does not expose `coveredCards` directly?
            # It exposes `calculateFinalRewards` which returns `rawPenalties`.
            # Let's rely on `calculateFinalRewards` for *component* scores if possible, 
            # OR just trust the C++ `calculateFinalRewards` if I updated it to include covered cards.
            # 
            # The User said: "Hard Fix: Ensure winner takes all penalties".
            # My C++ `calculateFinalRewards` might not do "Winner takes ALL". 
            # Let's implement it here in Python to be safe and explicit as requested.
            # But I cannot easily get covered cards if I didn't expose them.
            # 
            # Wait, I did update C++ `calculateFinalRewards` to include covered cards in `rawPenalties`.
            # So `res.rawPenalties` contains the correct cost for each player.
            pass
        
        # Get raw penalties from Engine (which now includes covered cards)
        res = game.calculateFinalRewards()
        raw_penalties = res.rawPenalties # vector<double> size 4 (index 0-3)
        
        # Determine Winner (Lowest Penalty)
        # Note: rawPenalties are positive sums of ranks.
        min_p = min(raw_penalties)
        winner_idx = -1
        
        # Handle Draw? Standard Sevens: usually one winner or share.
        # Let's assume one winner for "Winner Takes All".
        # If multiple have min_p, they split?
        # User says "Winner takes all penalties".
        
        total_pool = sum(raw_penalties)
        
        final_rewards = [0.0] * 4
        
        # Find winner(s)
        winners = [i for i, p in enumerate(raw_penalties) if p == min_p]
        
        if len(winners) == 1:
            w_idx = winners[0]
            # Winner gets (Total Pool - Own Score) ? Or Total Pool?
            # "Winner takes all penalties" usually means they get +Sum(Others).
            # And their own score is 0? Or they effectively "pay nothing" and get others?
            # If I have 0 penalty, I get Sum(Others).
            # If I have 5 penalty (but still lowest), do I get data?
            # Usually: Net = + (Sum of everyone else's penalty).
            # And Losers = - (Own Penalty).
            
            reward_pool = total_pool - raw_penalties[w_idx]
            final_rewards[w_idx] = reward_pool
            
            for i in range(4):
                if i != w_idx:
                    final_rewards[i] = -raw_penalties[i]
        else:
            # Draw logic (Split the pool?)
            # If Draw, maybe just -Own Score? Or Split positive?
            # Let's keep it simple: -Own Score for everyone, essentially no "Winner Bonus" transfer?
            # Or split the others?
            # Let's stick to: Everyone -Own Score. (Conservative)
            for i in range(4):
                final_rewards[i] = -raw_penalties[i]
                
        # Normalize
        SCALE = 100.0
        normalized_rewards = [r / SCALE for r in final_rewards]
        
        samples = []
        for s, pi, pid, b_targ in traj:
            z = normalized_rewards[pid - 1]
            samples.append((s, pi, b_targ, z))
            
        return samples

    else:
        # --- PYTHON PATH (Legacy/Fallback) ---
        game = game_cls()
        
        while not game.is_game_over():
            current_player = game.current_player_number
            state_tensor = game.get_state_tensor(current_player)
            
            # Model Inference
            # Add batch dim manually: (1, 11, 4, 13)
            inp = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                p_logits, _ = model(inp)
                
            # Logits -> Probs
            p_probs = torch.softmax(p_logits, dim=1).cpu().numpy().flatten()
            
            # Mask Invalid Moves
            valid_moves = game.get_all_valid_moves(current_player)
            legal_mask = np.zeros(52, dtype=np.float32)
            
            if len(valid_moves) == 0:
                # Should not happen in standard play (must cover)
                pass
            else:
                for card in valid_moves:
                    s, r = card.to_tensor_index()
                    idx = s * 13 + r
                    legal_mask[idx] = 1.0
                    
            # Apply mask
            p_probs = p_probs * legal_mask
            
            sum_p = p_probs.sum()
            if sum_p > 0:
                p_probs /= sum_p
            else:
                # Fallback to random legal
                if legal_mask.sum() > 0:
                    p_probs = legal_mask / legal_mask.sum()
                else:
                     # No moves? Game logic should handle this.
                    break 
            
            # Sample Action
            action_idx = np.random.choice(52, p=p_probs)
            
            # Decode action
            s_idx = action_idx // 13
            r_idx = action_idx % 13
            card = Card(s_idx + 1, r_idx + 1)
            
            # Record trajectory (Policy Target = Model Probabiltiy)
            # This reinforces the model's own preference but filtered by legality and outcome
            b_target = np.array(game.get_belief_target(current_player), dtype=np.float32).reshape(3, 4, 13)
            traj.append((state_tensor, p_probs, current_player, b_target))
            
            # Move
            game.make_move(card)
            game.next_player()
            
        # Rewards
        final_rewards = game.calculate_final_rewards()
        samples = []
        for s, pi, p_idx, b_targ in traj:
            z = final_rewards[p_idx - 1]
            samples.append((s, pi, b_targ, z))
            
        return samples

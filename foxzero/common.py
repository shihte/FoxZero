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
        
        return p, v

    def predict(self, state_tensor):
        """
        Inference wrapper.
        state_tensor: np.array [11, 4, 13]
        Returns: policy (np.array [52]), value (float)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_tensor).unsqueeze(0) # Add batch dim
            p, v = self(x)
            return p.squeeze(0).numpy(), v.item()

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
    
    game_samples = [] # (state, pi, player_idx)
    step_count = 0
    
    while not game.is_game_over():
        current_player = game.current_player_number
        
        # MCTS Search
        pi = mcts.search(game)
        state_tensor = game.get_state_tensor(current_player)
        
        game_samples.append((state_tensor, pi, current_player))
        
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
        for s, pi, p_idx in game_samples:
            z = final_rewards[p_idx - 1]
            final_samples.append((s, pi, z))
        return final_samples
    else:
        return []

def run_simulation_fast(game_cls, model):
    """
    Runs a simulation using direct Model Policy Sampling (No MCTS search).
    Optimized for Colab CPU workers where MCTS is too slow.
    Uses 'SevensGame' logic.
    """
    game = game_cls()
    model.eval()
    
    # Store trajectory
    traj = [] # (state_tensor, policy_probs, player_id)
    
    device = next(model.parameters()).device
    
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
            # But just in case
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
        traj.append((state_tensor, p_probs, current_player))
        
        # Move
        game.make_move(card)
        game.next_player()
        
    # Rewards
    final_rewards = game.calculate_final_rewards()
    samples = []
    for s, pi, p_idx in traj:
        z = final_rewards[p_idx - 1]
        samples.append((s, pi, z))
        
    return samples

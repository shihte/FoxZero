import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

from foxzero.game import SevensGame, Card
from foxzero.model import FoxZeroNet
from foxzero.mcts import MCTS

class FoxZeroDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # sample: (state, pi, reward)
        s, pi, z = self.samples[idx]
        return torch.FloatTensor(s), torch.FloatTensor(pi), torch.FloatTensor([z])

def self_play(net: FoxZeroNet, num_games=1, simulations=800, device='cpu') -> List[Tuple]:
    """
    Generates training data via self-play.
    Returns list of (state_tensor, pi, reward).
    """
    samples = []
    
    for game_idx in range(num_games):
        start_time = time.time()
        game = SevensGame()
        mcts = MCTS(net, simulations=simulations)
        
        game_samples = [] # [(state, pi, player_idx)]
        
        step_count = 0
        while not game.is_game_over():
            current_player = game.current_player_number
            
            # Get MCTS policy
            # For first 30 steps, temperature=1 (sampling). After, temp->0 (argmax)
            # Or just use the probabilities directly as target pi.
            # Usually we use temperature to CHOOSE the move, but target pi is visit count.
            # FoxZero design: "前 30 步: tau=1.0... 30 步後: tau->0".
            
            pi = mcts.search(game)
            state_tensor = game.get_state_tensor(current_player)
            
            # Store sample
            game_samples.append((state_tensor, pi, current_player))
            
            # Choose move
            valid_moves = game.get_all_valid_moves(current_player)
            
            if len(valid_moves) == 0:
                # Must be a pass
                 game.record_pass(current_player)
                 game.next_player()
                 continue
            
            # Sampling move
            if step_count < 30:
                # Sample from pi
                # Need to map pi back to valid moves?
                # pi is 52-dim. we just sample index.
                # However, pi might have small numerical noise on invalid moves?
                # MCTS logic ensured probability mass is only on valid moves.
                # Use numpy choice
                action_idx = np.random.choice(len(pi), p=pi)
            else:
                # Argmax
                action_idx = np.argmax(pi)
            
            # Convert index to card
            s_idx = action_idx // 13
            r_idx = action_idx % 13
            card = Card(s_idx + 1, r_idx + 1)
            
            # Execute move
            # Check validity (just in case)
            if card not in valid_moves:
                 # Fallback to random valid move if something weird happened with sampling
                 print(f"Warning: Selected invalid move {card}. Valid: {valid_moves}. Pi: {pi}")
                 card = np.random.choice(valid_moves)
            
            # Update game
            game.make_move(card)
            game.next_player()
            step_count += 1
            
            if step_count > 200:
                print("Game too long, aborting.")
                break
        
        if game.is_game_over():
            final_rewards = game.calculate_final_rewards()
            # Assign rewards to samples
            for s, pi, p_idx in game_samples:
                z = final_rewards[p_idx - 1]
                samples.append((s, pi, z))
                
            print(f"Game {game_idx+1} finished. Steps: {step_count}. Rewards: {final_rewards}. Duration: {time.time()-start_time:.1f}s")
        else:
            print(f"Game {game_idx+1} aborted.")
            
    return samples

def train(epochs=1, batch_size=32, num_games_per_epoch=1, simulations=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    net = FoxZeroNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        net.eval()
        
        # Self-Play
        print("Starting Self-Play...")
        samples = self_play(net, num_games=num_games_per_epoch, simulations=simulations, device=device)
        print(f"Generated {len(samples)} samples.")
        
        if len(samples) == 0:
            continue
            
        dataset = FoxZeroDataset(samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        net.train()
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        batches = 0
        
        for s_batch, pi_batch, z_batch in dataloader:
            s_batch = s_batch.to(device)
            pi_batch = pi_batch.to(device)
            z_batch = z_batch.to(device)
            
            optimizer.zero_grad()
            
            p_logits, v_pred = net(s_batch)
            
            # Loss Function:
            # L = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # v_pred is Tanh output [-1, 1]. z is [-1, 1].
            # p_logits are filtered? No, net outputs Raw Logits over 52 cards.
            # But MCTS pi is probability distribution.
            # We want CrossEntropy between target pi and predicted p.
            # Use LogSoftmax on p_logits -> NLLLoss with pi as weights?
            # Actually easiest is: loss = - sum(pi * log_softmax(p))
            
            log_probs = F.log_softmax(p_logits, dim=1)
            p_loss = -torch.sum(pi_batch * log_probs) / batch_size
            
            # Value loss (MSE)
            v_loss = F.mse_loss(v_pred, z_batch)
            
            loss = v_loss + p_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            batches += 1
            
        print(f"Epoch {epoch+1} Loss: {total_loss/batches:.4f} (P: {total_p_loss/batches:.4f}, V: {total_v_loss/batches:.4f})")
        
    print("Training complete.")
    torch.save(net.state_dict(), "models/foxzero_model.pth")
    print("Model saved to models/foxzero_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--sims", type=int, default=50)
    args = parser.parse_args()
    
    train(epochs=args.epochs, num_games_per_epoch=args.games, simulations=args.sims)

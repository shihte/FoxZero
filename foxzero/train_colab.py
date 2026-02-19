import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import numpy as np
from pathlib import Path

from foxzero.common import FoxZeroResNet, run_simulation_fast
from foxzero.game import SevensGame

# Configuration for Colab T4
# T4 has good FP32 performance, but Mixed Precision (AMP) is faster.
# CPU count is usually 2.
NUM_WORKERS = 2 
BATCH_SIZE = 512
# MCTS_SIMS not needed for fast mode

def colab_worker(queue, device, worker_id):
    """
    Worker process to generate self-play data.
    Runs on CPU using Fast Model Sampling (No MCTS).
    """
    print(f"Worker {worker_id} started on {device}")
    model = FoxZeroResNet().to(device)
    model.eval()
    
    # Track weight updates
    last_mod_time = 0
    weights_path = "foxzero_weights.pth"
    
    while True:
        # 1. Load latest weights if available
        if os.path.exists(weights_path):
            try:
                mod_time = os.path.getmtime(weights_path)
                if mod_time > last_mod_time:
                    # Load weights ensuring map_location is CPU
                    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)
                    last_mod_time = mod_time
                    # print(f"Worker {worker_id} loaded new weights.")
            except Exception as e:
                print(f"Worker {worker_id} failed to load weights: {e}")
                time.sleep(1)
        
        # 2. Run Simulation (Fast Mode)
        try:
            samples = run_simulation_fast(SevensGame, model)
            if len(samples) > 0:
                queue.put(samples)
        except Exception as e:
            print(f"Worker {worker_id} error in simulation: {e}")
            time.sleep(1)

def train_colab():
    mp.set_start_method('spawn', force=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Main Model
    model = FoxZeroResNet().to(device)
    model.train()
    
    # Load existing if available
    weights_path = "foxzero_weights.pth"
    if os.path.exists(weights_path):
        print("Loading existing weights...")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        # Initial save so workers can find it
        torch.save(model.state_dict(), weights_path)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() # Mixed precision
    
    # Queue for data
    queue = mp.Queue(maxsize=100) # Buffer of games
    
    # Start Workers
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=colab_worker, args=(queue, 'cpu', i))
        p.start()
        processes.append(p)
        
    print(f"Started {NUM_WORKERS} generator processes.")
    
    # Training Loop
    buffer = []
    total_steps = 0
    games_collected = 0
    
    try:
        while True:
            # 1. Collect Data
            # Try to get data from queue. If empty, wait a bit.
            # We want to fill buffer to at least BATCH_SIZE * 2
            while len(buffer) < BATCH_SIZE * 4:
                if not queue.empty():
                    game_samples = queue.get()
                    buffer.extend(game_samples)
                    games_collected += 1
                    if games_collected % 10 == 0:
                        print(f"Collected {games_collected} games. Buffer size: {len(buffer)}")
                else:
                    if len(buffer) == 0:
                         # Wait for initial data
                        time.sleep(1)
                    else:
                        break # Have some data, can train
            
            # 2. Prepare Batch
            # Sample random batch from buffer
            indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
            batch = [buffer[i] for i in indices]
            
            # Remove old data if buffer too big? 
            # Rolling window: Keep last 10000 samples
            if len(buffer) > 20000:
                buffer = buffer[-10000:]
                
            states, policies, values = zip(*batch)
            
            states_t = torch.stack(states).to(device)
            policies_t = torch.stack([torch.tensor(p) for p in policies]).to(device)
            values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(device)
            
            # 3. Optimization Step (AMP)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                p_logits, v_pred = model(states_t)
                
                # Loss
                # Policy: Cross Entropy
                # Value: MSE
                loss_p = torch.nn.functional.cross_entropy(p_logits, policies_t)
                loss_v = torch.nn.functional.mse_loss(v_pred, values_t)
                loss = loss_p + loss_v
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_steps += 1
            
            if total_steps % 10 == 0:
                print(f"Step {total_steps} | Loss: {loss.item():.4f} (P={loss_p.item():.4f}, V={loss_v.item():.4f}) | Buffer: {len(buffer)}")
                
            # 4. Save Weights
            if total_steps % 50 == 0:
                torch.save(model.state_dict(), weights_path)
                print(f"Saved weights to {weights_path}")
                
    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        for p in processes:
            p.terminate()
            p.join()
            
if __name__ == "__main__":
    train_colab()

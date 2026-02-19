import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path so we can import foxzero
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from foxzero.common import FoxZeroResNet, run_simulation_fast
from foxzero.game import SevensGame

# Configuration for Colab T4
# T4 has good FP32 performance, but Mixed Precision (AMP) is faster.
# CPU count is usually 2.
NUM_WORKERS = 2 
BATCH_SIZE = 512
# MCTS_SIMS not needed for fast mode

import argparse
import csv

# ... (Previous imports match, ensure mp, torch etc are there)

# Configuration
NUM_WORKERS = 2 
BATCH_SIZE = 512

def colab_worker(queue, device, worker_id, weights_path, temperature=1.0, dirichlet=None):
    """
    Worker process to generate self-play data.
    """
    print(f"Worker {worker_id} started on {device} (Temp={temperature}, Dirichlet={dirichlet})")
    model = FoxZeroResNet().to(device)
    model.eval()
    
    last_mod_time = 0
    
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
            except Exception as e:
                print(f"Worker {worker_id} failed to load weights: {e}")
                time.sleep(1)
        
        # 2. Run Simulation
        try:
            samples = run_simulation_fast(SevensGame, model, temperature=temperature, dirichlet_alpha=dirichlet)
            if len(samples) > 0:
                queue.put(samples)
        except Exception as e:
            print(f"Worker {worker_id} error in simulation: {e}")
            time.sleep(1)

def train_colab():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="foxzero_weights.pth")
    parser.add_argument("--log_path", type=str, default="train_log.csv")
    parser.add_argument("--temperature", type=float, default=1.0, help="Exploration temperature")
    parser.add_argument("--dirichlet", type=float, default=0.3, help="Dirichlet noise alpha (0 to disable)")
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    print(f"Weights Path: {args.weights_path}")
    print(f"Log Path: {args.log_path}")
    print(f"Exploration: Temp={args.temperature}, Dirichlet={args.dirichlet}")
    
    # Main Model
    model = FoxZeroResNet().to(device)
    model.train()
    
    # Load existing if available
    if os.path.exists(args.weights_path):
        print("Loading existing weights...")
        try:
            model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
        except:
            print("Failed to load weights, starting fresh.")
    else:
        # Initial save
        torch.save(model.state_dict(), args.weights_path)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Use new AMP API if available, else fallback
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        scaler = torch.cuda.amp.GradScaler()
    
    # Queue for data
    queue = mp.Queue(maxsize=100)
    
    # Start Workers
    processes = []
    
    # Handle Dirichlet Disable (0)
    d_alpha = args.dirichlet
    if d_alpha == 0: d_alpha = None
    
    for i in range(NUM_WORKERS):
        p = mp.Process(target=colab_worker, args=(queue, 'cpu', i, args.weights_path, args.temperature, d_alpha))
        p.start()
        processes.append(p)
        
    print(f"Started {NUM_WORKERS} generator processes.")
    
    # Init CSV
    if not os.path.exists(args.log_path):
        with open(args.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'loss_p', 'loss_v', 'buffer_size'])
            
    buffer = []
    total_steps = 0
    games_collected = 0
    
    # Resume step count from log if exists
    if os.path.exists(args.log_path):
        try:
            with open(args.log_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) > 0 and last_line[0].isdigit():
                        total_steps = int(last_line[0])
                        print(f"Resuming training from step {total_steps}")
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    try:
        while True:
            # 1. Collect Data
            # We strictly need at least BATCH_SIZE samples to train
            while len(buffer) < BATCH_SIZE:
                if not queue.empty():
                    game_samples = queue.get()
                    buffer.extend(game_samples)
                    games_collected += 1
                    if games_collected % 10 == 0:
                        print(f"Collected {games_collected} games. Buffer size: {len(buffer)}")
                else:
                    # Queue empty, wait for workers
                    time.sleep(1)
            
            # Opportunistically collect more if available (up to 4x batch)
            while not queue.empty() and len(buffer) < BATCH_SIZE * 4:
                game_samples = queue.get()
                buffer.extend(game_samples)
                games_collected += 1
            
            # 2. Prepare Batch
            indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
            batch = [buffer[i] for i in indices]
            
            if len(buffer) > 20000:
                buffer = buffer[-5000:] # Trim aggressively to keep fresh data
                
            states, policies, values = zip(*batch)
            
            states_t = torch.stack([torch.from_numpy(s) for s in states]).to(device)
            policies_t = torch.stack([torch.tensor(p) for p in policies]).to(device)
            values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(device)
            
            # 3. Optimization
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                p_logits, v_pred = model(states_t)
                loss_p = torch.nn.functional.cross_entropy(p_logits, policies_t)
                loss_v = torch.nn.functional.mse_loss(v_pred, values_t)
                loss = loss_p + loss_v
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_steps += 1
            
            # Logging
            if total_steps % 10 == 0:
                print(f"Step {total_steps} | Loss: {loss.item():.4f} | Buffer: {len(buffer)}")
                with open(args.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([total_steps, loss.item(), loss_p.item(), loss_v.item(), len(buffer)])
                
            # Save Weights
            if total_steps % 50 == 0:
                torch.save(model.state_dict(), args.weights_path)
                print(f"Saved weights to {args.weights_path}")
                
    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    train_colab()

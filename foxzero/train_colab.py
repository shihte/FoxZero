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
NUM_WORKERS = 2 
BATCH_SIZE = 512

import argparse
import csv

def get_curriculum_params(step):
    """
    Returns (temperature, dirichlet_alpha, lr, freeze_backbone)
    based on the 300k-step 4-phase schedule.
    """
    if step < 150000:
        # Phase 1: Base Internalization
        return 1.0, None, 1e-4, False
    elif step < 210000:
        # Phase 2: Exploration
        progress = (step - 150000) / 60000.0
        temp = 1.5 - (0.3 * progress) # Anneal from 1.5 to 1.2
        return temp, 0.3, 1e-4, False
    elif step < 260000:
        # Phase 3: Exploitation
        progress = (step - 210000) / 50000.0
        temp = 0.8 - (0.3 * progress) # Anneal from 0.8 to 0.5
        return temp, None, 1e-5, False
    else:
        # Phase 4: Belief Head Tuning
        progress = (step - 260000) / 40000.0
        progress = min(1.0, progress)
        temp = 0.5 - (0.4 * progress) # Anneal to 0.1
        return temp, None, 1e-5, True

def colab_worker(queue, device, worker_id, weights_path, shared_step, top_k=None):
    """
    Worker process to generate self-play data using dynamic curriculum.
    """
    print(f"Worker {worker_id} started on {device}.")
    model = FoxZeroResNet().to(device)
    model.eval()
    
    last_mod_time = 0
    
    while True:
        # 1. Load latest weights if available
        if os.path.exists(weights_path):
            try:
                mod_time = os.path.getmtime(weights_path)
                if mod_time > last_mod_time:
                    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)
                    last_mod_time = mod_time
            except Exception as e:
                print(f"Worker {worker_id} failed to load weights: {e}")
                time.sleep(1)
        
        # 2. Get current curriculum parameters
        current_step = shared_step.value
        temp, dirichlet, _, _ = get_curriculum_params(current_step)
        
        # 3. Run Simulation
        try:
            samples = run_simulation_fast(SevensGame, model, temperature=temp, dirichlet_alpha=dirichlet, top_k=top_k)
            if len(samples) > 0:
                queue.put(samples)
        except Exception as e:
            print(f"Worker {worker_id} error in simulation: {e}")
            time.sleep(1)

def train_colab():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="models/foxzero_weights.pth")
    parser.add_argument("--log_path", type=str, default="logs/train_log.csv")
    parser.add_argument("--top_k", type=int, default=0, help="Top-K sampling (0 to disable)")
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    print(f"Weights Path: {args.weights_path}")
    print(f"Log Path: {args.log_path}")
    print("Using 300k-step 4-Phase Curriculum Manager")
    
    # Main Model
    model = FoxZeroResNet().to(device)
    model.train()
    
    if os.path.exists(args.weights_path):
        print("Loading existing weights...")
        try:
            model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
        except:
            print("Failed to load weights, starting fresh.")
    else:
        torch.save(model.state_dict(), args.weights_path)
        
    # Setup Parameter Groups for freezing in Phase 4
    backbone_params = []
    belief_params = []
    for name, p in model.named_parameters():
        if 'belief' in name:
            belief_params.append(p)
        else:
            backbone_params.append(p)
            
    optimizer = torch.optim.Adam([
        {'params': backbone_params},
        {'params': belief_params}
    ], lr=1e-4, weight_decay=1e-4)
    
    try:
        scaler = torch.amp.GradScaler('cuda')
    except:
        scaler = torch.cuda.amp.GradScaler()
    
    queue = mp.Queue(maxsize=100)
    
    total_steps = 0
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
            
    # Shared step counter for workers
    shared_step = mp.Value('i', total_steps)
    
    k = args.top_k if args.top_k > 0 else None
    
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=colab_worker, args=(queue, 'cpu', i, args.weights_path, shared_step, k))
        p.start()
        processes.append(p)
        
    print(f"Started {NUM_WORKERS} generator processes.")
    
    if not os.path.exists(args.log_path):
        with open(args.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'loss_p', 'loss_v', 'loss_b', 'buffer_size', 'temp', 'lr', 'ups'])
            
    buffer = []
    games_collected = 0
    
    try:
        while True:
            # 1. Collect Data (Ingest everything from queue)
            while not queue.empty():
                game_samples = queue.get()
                buffer.extend(game_samples)
                games_collected += 1
                if games_collected % 10 == 0:
                    print(f"Collected {games_collected} games. Buffer size: {len(buffer)}")
            
            # Wait until we have at least one batch
            if len(buffer) < BATCH_SIZE:
                time.sleep(1)
                continue
            
            # Sliding Window: Keep only the latest 20,000 samples
            if len(buffer) > 20000:
                buffer = buffer[-20000:]
            
            # 2. Prepare Batch
            indices = np.random.choice(len(buffer), BATCH_SIZE, replace=False)
            batch = [buffer[i] for i in indices]
                
            states, policies, b_targets, values = zip(*batch)
            
            states_t = torch.stack([torch.from_numpy(s) for s in states]).to(device)
            policies_t = torch.stack([torch.tensor(p) for p in policies]).to(device)
            b_targets_t = torch.stack([torch.tensor(b) for b in b_targets]).to(device)
            values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(device)
            
            # Audit labels (Addressing User Doubt)
            if total_steps % 10 == 0:
                avg_b_sum = b_targets_t.sum().item() / BATCH_SIZE
                print(f"DEBUG | Step {total_steps} | Belief Target Sum(avg): {avg_b_sum:.2f} (Expected 10~39)")
            
            # 3. Update Curriculum Parameters
            shared_step.value = total_steps
            temp, dirichlet, lr, freeze_backbone = get_curriculum_params(total_steps)
            
            optimizer.param_groups[0]['lr'] = 1e-6 if freeze_backbone else lr # Backbone
            optimizer.param_groups[1]['lr'] = lr # Belief Head
            
            # 4. Optimization
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                p_logits, v_pred, b_pred = model(states_t)
                loss_p = torch.nn.functional.cross_entropy(p_logits, policies_t)
                loss_v = torch.nn.functional.mse_loss(v_pred, values_t)
                loss_b = torch.nn.functional.binary_cross_entropy_with_logits(b_pred, b_targets_t)
                
                loss = loss_p + loss_v + (0.5 * loss_b)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_steps += 1
            
            # 5. Logging
            if total_steps % 10 == 0:
                print(f"Step {total_steps} | L: {loss.item():.3f} (P:{loss_p.item():.3f} V:{loss_v.item():.3f} B:{loss_b.item():.3f}) | Buffer: {len(buffer)} | T: {temp:.2f}")
                
                # Check if file is empty to write header
                file_empty = not os.path.exists(args.log_path) or os.path.getsize(args.log_path) == 0
                with open(args.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if file_empty:
                        writer.writerow(['step', 'loss', 'loss_p', 'loss_v', 'loss_b', 'buffer_size', 'temp', 'lr', 'ups'])
                    writer.writerow([total_steps, loss.item(), loss_p.item(), loss_v.item(), loss_b.item(), len(buffer), temp, lr, 0.0])
                
            # 6. Save Weights
            if total_steps % 100 == 0:
                torch.save(model.state_dict(), args.weights_path)
                print(f"[SAVE] 訓練達 {total_steps} 步，權重已自動儲存至: {args.weights_path}")
                
    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    train_colab()

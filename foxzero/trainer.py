import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import glob
from collections import deque
from foxzero.common import FoxZeroResNet

# Constants
WEIGHTS_PATH = "models/foxzero_weights.pth"
DATA_POOL_DIR = "data_pool"

def train(args):
    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (Fallback)")
        
    # 2. Model Setup
    model = FoxZeroResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Fixed LR for now
    
    # Initial Weights Save
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Initialized weights at {WEIGHTS_PATH}")
    
    # 3. Replay Buffer
    replay_buffer = deque(maxlen=50000)
    
    updates = 0
    start_time = time.time()
    
    # Ensure data pool exists
    os.makedirs(DATA_POOL_DIR, exist_ok=True)
    
    print("Waiting for data...")
    
    while True:
        # 4. Data Ingestion
        # Scan for .pt files
        files = glob.glob(os.path.join(DATA_POOL_DIR, "*.pt"))
        
        if files:
            for filepath in files:
                try:
                    # Load
                    # We use map_location=cpu to avoid overloading GPU memory during load
                    batch_samples = torch.load(filepath, map_location='cpu', weights_only=False)
                    
                    # Add to buffer
                    for sample in batch_samples:
                        replay_buffer.append(sample)
                        
                    # Delete file immediately
                    os.remove(filepath)
                    
                except Exception as e:
                    print(f"Error reading/deleting {filepath}: {e}")
                    # Try to remove corrupt file?
                    try:
                        os.remove(filepath)
                    except:
                        pass
        
        # 5. Check Training Condition
        if len(replay_buffer) < args.batch_size:
            # Sleep if idle and buffer low
            if not files:
                time.sleep(1.0)
            continue
            
        # 6. Training Step
        indices = np.random.choice(len(replay_buffer), args.batch_size, replace=False)
        
        s_batch, pi_batch, z_batch = [], [], []
        for idx in indices:
            s, pi, z = replay_buffer[idx]
            s_batch.append(s)
            pi_batch.append(pi)
            z_batch.append(z)
            
        s_tensor = torch.FloatTensor(np.array(s_batch)).to(device)
        pi_tensor = torch.FloatTensor(np.array(pi_batch)).to(device)
        z_tensor = torch.FloatTensor(np.array(z_batch)).unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        p_logits, v_pred = model(s_tensor)
        
        log_probs = F.log_softmax(p_logits, dim=1)
        p_loss = -torch.sum(pi_tensor * log_probs) / args.batch_size
        v_loss = F.mse_loss(v_pred, z_tensor)
        
        loss = p_loss + v_loss
        loss.backward()
        optimizer.step()
        
        updates += 1
        
        # 7. Logging & Saving
        if updates % 10 == 0:
            elapsed = time.time() - start_time
            rate = updates / elapsed if elapsed > 0 else 0
            print(f"[Trainer] Update {updates} | Loss: {loss.item():.4f} | Buffer: {len(replay_buffer)} | Rate: {rate:.2f} ups")
            
        if updates % 100 == 0:
             # Atomic save?
             # Save to tmp then rename to ensure Generator doesn't read partial file
             tmp_path = WEIGHTS_PATH + ".tmp"
             torch.save(model.state_dict(), tmp_path)
             os.rename(tmp_path, WEIGHTS_PATH)
             # print(f"Saved weights to {WEIGHTS_PATH}")
             
        if args.max_updates > 0 and updates >= args.max_updates:
            print("Max updates reached.")
            torch.save(model.state_dict(), WEIGHTS_PATH)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_updates", type=int, default=10000)
    args = parser.parse_args()
    
    train(args)

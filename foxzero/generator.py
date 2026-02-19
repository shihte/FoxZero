import torch
import torch.multiprocessing as mp
import os
import time
import argparse
import uuid
from pathlib import Path
from foxzero.common import FoxZeroResNet, run_mcts_game_simulation
from foxzero.game import SevensGame

# Constants
WEIGHTS_PATH = "foxzero_weights.pth"
DATA_POOL_DIR = "data_pool"

def actor_worker(rank, args):
    """
    Actor Worker Process (CPU)
    """
    # Force CPU
    device = torch.device('cpu') 
    torch.set_num_threads(1) # Important for multiprocessing efficiency
    
    # Seeds
    seed = int(time.time()) + rank
    torch.manual_seed(seed)
    
    print(f"[Actor {rank}] Started on CPU. PID: {os.getpid()}")
    
    # Initialize Model
    model = FoxZeroResNet()
    model.eval()
    
    # Track weights
    last_mod_time = 0
    
    while True:
        # 1. Check for new weights
        if os.path.exists(WEIGHTS_PATH):
            try:
                mod_time = os.path.getmtime(WEIGHTS_PATH)
                if mod_time > last_mod_time:
                    # Reload
                    # Use a lock file or just try-except? 
                    # The trainer writes atomically? Ideally yes.
                    # Trainer should write to tmp then rename.
                    # We assume it does or we just retry.
                    try:
                        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
                        model.load_state_dict(state_dict)
                        last_mod_time = mod_time
                        if rank == 0:
                            print(f"[Actor {rank}] Loaded new weights from {WEIGHTS_PATH}")
                    except Exception as e:
                        print(f"[Actor {rank}] Failed to load weights: {e}")
                        time.sleep(1)
            except OSError:
                pass

        # 2. Run Simulation
        # run_mcts_game_simulation(game_cls, model, sims)
        samples = run_mcts_game_simulation(SevensGame, model, args.sims)
        
        if len(samples) > 0:
            # 3. Save Data
            # Format: batch_{timestamp}_{uuid}.pt
            timestamp = int(time.time() * 1000)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"batch_{timestamp}_{rank}_{unique_id}.pt"
            filepath = os.path.join(DATA_POOL_DIR, filename)
            
            try:
                # Save as list of tuples (or you can collate here, but list is finer)
                torch.save(samples, filepath)
                # print(f"[Actor {rank}] Saved {filepath} ({len(samples)} samples)")
            except Exception as e:
                print(f"[Actor {rank}] Failed to save data: {e}")
        
        # Check if we should stop? No, infinite loop.

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=6)
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()
    
    # Ensure data pool exists
    os.makedirs(DATA_POOL_DIR, exist_ok=True)
    
    print(f"Starting {args.actors} Actors...")
    print(f"Simulations: {args.sims}")
    print(f"Data Pool: {os.path.abspath(DATA_POOL_DIR)}")
    
    processes = []
    for i in range(args.actors):
        p = mp.Process(target=actor_worker, args=(i, args))
        p.start()
        processes.append(p)
        
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping Generators...")
        for p in processes:
            p.terminate()

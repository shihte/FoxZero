import torch
import torch.multiprocessing as mp
import numpy as np
import os
import time
import argparse
import uuid
from pathlib import Path
from foxzero.common import FoxZeroResNet, run_mcts_game_simulation, run_simulation_fast
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
    np.random.seed(seed)
    
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
                    try:
                        # Load to CPU to avoid CUDA error on actor
                        # We use map_location='cpu' which is correct
                        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
                        # Handle potential DDP prefix if trained on multi-GPU (though T4 usually single)
                        # but good practice to remove 'module.' prefix just in case
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith('module.'):
                                new_state_dict[k[7:]] = v
                            else:
                                new_state_dict[k] = v
                                
                        model.load_state_dict(new_state_dict)
                        last_mod_time = mod_time
                        if rank == 0:
                            print(f"[Actor {rank}] Loaded new weights from {WEIGHTS_PATH}")
                    except Exception as e:
                        print(f"[Actor {rank}] Failed to load weights (retrying): {e}")
            except OSError:
                pass

        # 2. Run Simulation (FAST POLICY SAMPLING)
        # Replaced MCTS with run_simulation_fast for Colab T4 optimization
        # This provides 10x speedup
        # Extract exploration params
        temp = args.temperature
        alpha = args.dirichlet
        if alpha == 0: alpha = None
        
        samples = run_simulation_fast(SevensGame, model, temperature=temp, dirichlet_alpha=alpha)
        
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
    parser.add_argument("--sims", type=int, default=200) # Not used in fast mode but kept for compat
    parser.add_argument("--temperature", type=float, default=1.0, help="Exploration temperature")
    parser.add_argument("--dirichlet", type=float, default=0.3, help="Dirichlet noise alpha (0 to disable)")
    args = parser.parse_args()
    
    # Ensure data pool exists
    import numpy as np # Ensure numpy is imported in main
    os.makedirs(DATA_POOL_DIR, exist_ok=True)
    
    print(f"Starting {args.actors} Actors...")
    print(f"Exploration: Temp={args.temperature}, Dirichlet={args.dirichlet}")
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

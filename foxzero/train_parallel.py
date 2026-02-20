import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import queue
from collections import deque
from foxzero.game import SevensGame, Card
from foxzero.model import FoxZeroNet
from foxzero.mcts import MCTS

# Configuration
MODEL_PATH = "foxzero_parallel_model.pth"
CHECKPOINT_INTERVAL = 50  # Save model every X updates
ACTOR_SYNC_INTERVAL = 10  # Actor reloads model every X games

def actor_process(rank, data_queue, args):
    """
    Actor Process:
    - Identifies as rank.
    - Loads model (CPU).
    - Runs Self-Play.
    - Sends samples to Learner via Queue.
    """
    # Set seed for reproducibility/variance
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    print(f"[Actor {rank}] Started.")
    
    # Initialize Model (CPU)
    net = FoxZeroNet()
    net.eval()
    
    # Wait for initial model to be saved by Learner
    while not os.path.exists(MODEL_PATH):
        time.sleep(1)
    
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    except Exception as e:
        print(f"[Actor {rank}] Error loading model: {e}")
        
    game_count = 0
    samples_produced = 0
    
    while True:
        # Sync Model Weights periodically
        if game_count % ACTOR_SYNC_INTERVAL == 0 and game_count > 0:
            try:
                # We use a try-except block because the learner might be writing to the file
                # A robust way is atomic rename, but simple load usually works if file system is good.
                net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            except Exception:
                pass # Retry next time
        
        # Run 1 Game
        game = SevensGame()
        mcts = MCTS(net, simulations=args.sims)
        
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
                
            # Sampling vs Argmax
            # For training, we usually sample early on.
            if step_count < 30:
                action_idx = np.random.choice(len(pi), p=pi)
            else:
                action_idx = np.argmax(pi)
            
            # Convert to Card
            s_idx = action_idx // 13
            r_idx = action_idx % 13
            card = Card(s_idx + 1, r_idx + 1)
            
            if card not in valid_moves:
                card = valid_moves[0] # Fallback
                
            game.make_move(card)
            game.next_player()
            step_count += 1
            if step_count > 300: break
            
        # Process result
        if game.is_game_over():
            final_rewards = game.calculate_final_rewards()
            # Send samples to Learner
            # Prepare batch of samples
            data_batch = []
            for s, pi, p_idx in game_samples:
                z = final_rewards[p_idx - 1]
                data_batch.append((s, pi, z))
            
            # Put in Queue
            data_queue.put(data_batch)
            samples_produced += len(data_batch)
            game_count += 1
            
            if game_count % 10 == 0:
                 print(f"[Actor {rank}] Generated {game_count} games ({samples_produced} samples).")
        else:
            print(f"[Actor {rank}] Game aborted.")

def learner_process(data_queue, args):
    """
    Learner Process:
    - Initializes Model (GPU/CPU).
    - Consumes data from Queue.
    - Updates Model.
    - Saves Checkpoint.
    """
    device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
        
    print(f"[Learner] Started on {device}.")
    
    net = FoxZeroNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Save Initial Model
    torch.save(net.state_dict(), MODEL_PATH)
    
    replay_buffer = deque(maxlen=args.buffer_size)
    
    steps = 0
    updates = 0
    start_time = time.time()
    
    while True:
        # 1. Collect Data
        try:
            # Non-blocking get with limit
            # Fetch as many as possible?
            while True:
                batch = data_queue.get_nowait()
                for sample in batch:
                    replay_buffer.append(sample)
        except queue.Empty:
            pass
        
        # 2. Check if we can train
        if len(replay_buffer) < args.batch_size:
            time.sleep(0.1)
            continue
            
        # 3. Train Step
        # Sample batch
        indices = np.random.choice(len(replay_buffer), args.batch_size, replace=False)
        
        s_batch_list, pi_batch_list, z_batch_list = [], [], []
        for idx in indices:
            s, pi, z = replay_buffer[idx]
            s_batch_list.append(s)
            pi_batch_list.append(pi)
            z_batch_list.append(z)
            
        s_tensor = torch.FloatTensor(np.array(s_batch_list)).to(device)
        pi_tensor = torch.FloatTensor(np.array(pi_batch_list)).to(device)
        z_tensor = torch.FloatTensor(np.array(z_batch_list)).unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        p_logits, v_pred = net(s_tensor)
        
        log_probs = F.log_softmax(p_logits, dim=1)
        p_loss = -torch.sum(pi_tensor * log_probs) / args.batch_size
        v_loss = F.mse_loss(v_pred, z_tensor)
        
        loss = p_loss + v_loss
        loss.backward()
        optimizer.step()
        
        updates += 1
        
        # 4. Checkpoint
        if updates % CHECKPOINT_INTERVAL == 0:
            torch.save(net.state_dict(), MODEL_PATH)
            
        if updates % 10 == 0:
            elapsed = time.time() - start_time
            rate = updates/elapsed
            print(f"[Learner] Update {updates} | Loss: {loss.item():.4f} | Buffer: {len(replay_buffer)} | Rate: {rate:.2f} ups")
            
            # CSV Logging
            if args.log_file:
                with open(args.log_file, "a") as f:
                    f.write(f"{updates},{loss.item():.6f},{p_loss.item():.6f},{v_loss.item():.6f},{len(replay_buffer)},{rate:.2f}\n")
            
        if args.max_updates > 0 and updates >= args.max_updates:
            print("[Learner] Max updates reached. Exiting.")
            torch.save(net.state_dict(), MODEL_PATH)
            break

if __name__ == "__main__":
    # Required for MacOS/PyTorch sharing
    mp.set_start_method('spawn', force=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=4, help="Number of actor processes")
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="logs/train_log.csv", help="Path to CSV log file")
    args = parser.parse_args()
    
    # Initialize Log File
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write("update,total_loss,policy_loss,value_loss,buffer_size,ups\n")
    
    # Shared Queue
    # We use a Queue to send games from Actor to Learner
    data_queue = mp.Queue(maxsize=100) # Prevents actors from overproducing if learner is slow
    
    processes = []
    
    # Start Learner
    p_learner = mp.Process(target=learner_process, args=(data_queue, args))
    p_learner.start()
    processes.append(p_learner)
    
    # Start Actors
    for i in range(args.actors):
        p_actor = mp.Process(target=actor_process, args=(i, data_queue, args))
        p_actor.start()
        processes.append(p_actor)
        
    try:
        p_learner.join() # Wait for learner to finish (max_updates)
    except KeyboardInterrupt:
        print("Stopping...")
        
    for p in processes:
        p.terminate()
        p.join()

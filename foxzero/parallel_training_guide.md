# FoxZero Parallel Training Guide

This guide details the parameters for `foxzero/train_parallel.py` and provides recommended configurations for different hardware setups.

## Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--actors` | `int` | `4` | Number of parallel CPU processes to run. Each process runs its own game loop and MCTS. |
| `--sims` | `int` | `800` | Number of MCTS simulations per move. Higher = Stronger play but slower generation. |
| `--batch_size` | `int` | `64` | Number of samples per training step. |
| `--buffer_size` | `int` | `10000` | Max number of samples in the Replay Buffer. Older samples are discarded. |
| `--lr` | `float` | `0.001` | Learning Rate for the Adam optimizer. |
| `--max_updates` | `int` | `1000` | Total number of training steps (updates) to perform before stopping. |
| `--log_file` | `str` | `"train_log.csv"` | Path to save training metrics (CSV format). |

## Recommended Configurations

### 1. Apple M1/M2/M3 (MacBook Pro)
Since M-series chips have unified memory and a limited number of performance cores, you should match `--actors` to your **Performance Cores**, not total cores.

- **Command**:
```bash
PYTHONPATH=. python3 foxzero/train_parallel.py --actors 4 --sims 400 --batch_size 64 --max_updates 5000
```
- **Reasoning**:
    - `actors=4`: Leaves resources for the OS and the Learner (GPU) process.
    - `sims=400`: 800 might be too slow on CPU; 400 is a good balance for dev/debugging.

### 2. High-End Workstation (e.g. Threadripper + RTX 4090)
High core count allows for massive parallelism.

- **Command**:
```bash
PYTHONPATH=. python3 foxzero/train_parallel.py --actors 32 --sims 800 --batch_size 256 --buffer_size 50000 --max_updates 20000
```

### 3. Data Center / H100 (Cloud)
If you have massive CPU resources (e.g. 64+ cores) and a powerful H100 GPU.

- **Command**:
```bash
PYTHONPATH=. python3 foxzero/train_parallel.py --actors 60 --sims 1600 --batch_size 512 --buffer_size 100000 --max_updates 100000
```
- **Reasoning**:
    - `actors=60`: Saturation of CPU cores.
    - `sims=1600`: Standard AlphaZero count for high-quality play.
    - `batch_size=512`: H100 optimization.

## Tuning Tips

1.  **Monitor GPU Utilization**: If GPU utilization is low (< 20%), increase `--actors`. If you run out of CPU cores, decrease `--sims`.
2.  **Buffer Size**: Ensure `buffer_size` is large enough to hold samples from all actors for a few minutes of play, to prevent overfitting to the most recent game.
3.  **Synchronization**: The script syncs weights every 10 games. If training is very fast, the actors might lag behind. This is generally fine for AlphaZero (asynchronous).

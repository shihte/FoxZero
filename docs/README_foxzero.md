# FoxZero (Python Implementation)

FoxZero is an AlphaZero-style reinforcement learning agent for the card game Sevens (排七).

## Project Structure

- `foxzero/`: Core Python package.
    - `game.py`: Game logic and State Encoding.
    - `model.py`: Neural Network Architecture (PyTorch).
    - `mcts.py`: Monte Carlo Tree Search.
    - `train.py`: Self-Play and Training Loop.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Numpy

### Running Training
To train the agent:

```bash
python3 -m foxzero.train --epochs 10 --games 10 --sims 800
```

### Running Tests
```bash
python3 -m tests.test_game
python3 -m tests.test_model
python3 -m tests.test_mcts
```

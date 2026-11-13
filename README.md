# Neural Snake

An end-to-end Snake project where a Deep Q-learning agent learns to find food and avoid collisions.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`

## Setup

Open a terminal in this project folder, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Training defaults to 1,000 games. The Snake window shows the current game, and the terminal prints each game's score and average. You can close the game window to stop early. For a shorter run or a slower display:

```bash
python train.py --episodes 200 --speed 30
```

The best and latest network weights are saved in `models/`. A score chart is saved to `outputs/training_scores.png`.

## Watch the trained agent

```bash
python play.py
```

To use another saved model or adjust playback speed:

```bash
python play.py --model models/latest_model.pth --speed 8
```

## Project files

- `game.py` contains the playable Snake environment, movement, collision rules, food, reward, and rendering.
- `agent.py` encodes the game as 11 inputs and implements exploration, replay memory, and Q-learning.
- `model.py` defines a neural network with 11 inputs, a 256-neuron hidden layer, and 3 action scores.
- `helper.py` saves the training score chart.
- `train.py` runs games and updates the agent.
- `play.py` loads saved weights and lets the agent play without exploration.
- `requirements.txt` lists Python dependencies.

The three actions are straight, turn right, and turn left relative to the snake's current direction. The agent receives a positive reward for eating food, a negative reward for a crash, and a small step penalty to encourage reaching food efficiently.

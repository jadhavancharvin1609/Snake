"""Watch a trained Snake agent play."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from game import SnakeGame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best_model.pth", help="saved model file")
    parser.add_argument("--speed", type=int, default=12, help="display speed in frames per second")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"model not found: {model_path}; train an agent first with python train.py")

    agent = Agent()
    agent.load(model_path)
    game = SnakeGame(speed=args.speed)
    try:
        while not game.quit_requested:
            state = agent.get_state(game)
            action = agent.get_action(state, explore=False)
            _, done, _ = game.play_step(action.index(1))
            if done and not game.quit_requested:
                game.reset()
    finally:
        game.close()


if __name__ == "__main__":
    main()

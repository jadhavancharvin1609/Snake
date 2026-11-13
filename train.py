"""Train the Snake agent with Deep Q-learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from game import SnakeGame
from helper import plot_scores


def train(episodes: int, speed: int, seed: int | None) -> None:
    agent = Agent(seed=seed)
    game = SnakeGame(speed=speed)
    scores: list[int] = []
    mean_scores: list[float] = []
    total_score = 0
    best_score = 0

    try:
        while agent.n_games < episodes and not game.quit_requested:
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, done, score = game.play_step(action.index(1))
            next_state = agent.get_state(game)

            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
                if score > best_score:
                    best_score = score
                    agent.save("models/best_model.pth")
                agent.save("models/latest_model.pth")
                plot_scores(scores, mean_scores, "outputs/training_scores.png")
                print(f"Game {agent.n_games:4d}/{episodes} | score {score:3d} | "
                      f"average {mean_score:5.2f} | best {best_score:3d}")
    finally:
        game.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000, help="number of games to train")
    parser.add_argument("--speed", type=int, default=80, help="game display speed in frames per second")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed for repeatable runs")
    args = parser.parse_args()
    if args.episodes < 1 or args.speed < 1:
        parser.error("episodes and speed must be positive")
    train(args.episodes, args.speed, args.seed)


if __name__ == "__main__":
    main()

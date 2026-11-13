"""Deep Q-learning agent and replay memory for Snake."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from game import SnakeGame
from model import LinearQNet


class Agent:
    def __init__(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.n_games = 0
        self.epsilon = 0.0
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=100_000)
        self.batch_size = 1_000
        self.model = LinearQNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def get_state(game: SnakeGame) -> np.ndarray:
        """Encode danger, direction, and food location as 11 boolean inputs."""
        head = game.snake[0]
        direction_index = game.direction_index
        straight = SnakeGame.DIRECTIONS[direction_index]
        right = SnakeGame.DIRECTIONS[(direction_index + 1) % 4]
        left = SnakeGame.DIRECTIONS[(direction_index - 1) % 4]

        def next_point(vector: tuple[int, int]) -> tuple[int, int]:
            return head[0] + vector[0], head[1] + vector[1]

        def danger(point: tuple[int, int]) -> bool:
            x, y = point
            outside = x < 0 or x >= game.columns or y < 0 or y >= game.rows
            return outside or point in game.snake[:-1]

        food_x, food_y = game.food
        return np.array([
            danger(next_point(straight)),
            danger(next_point(right)),
            danger(next_point(left)),
            direction_index == 3,  # moving up
            direction_index == 1,  # moving down
            direction_index == 2,  # moving left
            direction_index == 0,  # moving right
            food_x < head[0],
            food_x > head[0],
            food_y < head[1],
            food_y > head[1],
        ], dtype=np.float32)

    def remember(self, state: np.ndarray, action: list[int], reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, int(np.argmax(action)), reward, next_state, done))

    def train_short_memory(self, state: np.ndarray, action: list[int], reward: float,
                           next_state: np.ndarray, done: bool) -> None:
        self._train_batch([(state, int(np.argmax(action)), reward, next_state, done)])

    def train_long_memory(self) -> None:
        if not self.memory:
            return
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        self._train_batch(batch)

    def _train_batch(self, batch: list[tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        states = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([item[3] for item in batch]), dtype=torch.float32)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.bool)

        current_q = self.model(states)
        target_q = current_q.detach().clone()
        with torch.no_grad():
            next_q_max = self.model(next_states).max(dim=1).values
        target_values = rewards + self.gamma * next_q_max * (~dones)
        target_q[torch.arange(len(batch)), actions] = target_values

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state: np.ndarray, explore: bool = True) -> list[int]:
        """Return a one-hot action: straight, right, or left."""
        self.epsilon = max(0, 80 - self.n_games) if explore else 0
        if explore and random.randrange(200) < self.epsilon:
            move = random.randrange(3)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
            move = int(torch.argmax(q_values).item())
        action = [0, 0, 0]
        action[move] = 1
        return action

    def save(self, path: str | Path) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval()

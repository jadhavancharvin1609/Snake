"""Small fully connected neural network used to score Snake actions."""

from __future__ import annotations

import torch
from torch import nn


class LinearQNet(nn.Module):
    """Maps an 11-value game state to one Q-value per relative action."""

    def __init__(self, input_size: int = 11, hidden_size: int = 256,
                 output_size: int = 3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

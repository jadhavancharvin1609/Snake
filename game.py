"""Snake game environment used by the training and play programs."""

from __future__ import annotations

import random

import pygame


Point = tuple[int, int]


class SnakeGame:
    """A small grid-based Snake game with a step-based agent interface."""

    WHITE = (245, 245, 245)
    BLACK = (24, 24, 28)
    GREEN = (40, 190, 90)
    DARK_GREEN = (25, 135, 65)
    RED = (220, 65, 65)
    GRID = (45, 45, 50)

    # Coordinates use x to the right and y downward.
    DIRECTIONS: tuple[Point, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(self, width: int = 640, height: int = 480, cell_size: int = 20,
                 speed: int = 80, show_window: bool = True) -> None:
        if width % cell_size or height % cell_size:
            raise ValueError("width and height must be multiples of cell_size")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.columns = width // cell_size
        self.rows = height // cell_size
        self.speed = speed
        self.show_window = show_window

        pygame.init()
        self.screen = pygame.display.set_mode((width, height)) if show_window else None
        if show_window:
            pygame.display.set_caption("Neural Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.quit_requested = False
        self.reset()

    def reset(self) -> None:
        """Start a fresh game and return the snake to the center of the board."""
        center = (self.columns // 2, self.rows // 2)
        self.snake: list[Point] = [center, (center[0] - 1, center[1]), (center[0] - 2, center[1])]
        self.direction_index = 0  # right
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0

    def _place_food(self) -> Point:
        empty_cells = [
            (x, y)
            for x in range(self.columns)
            for y in range(self.rows)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            return self.snake[0]
        return random.choice(empty_cells)

    def _is_collision(self, point: Point) -> bool:
        x, y = point
        return (
            x < 0 or x >= self.columns or y < 0 or y >= self.rows
            or point in self.snake[:-1]
        )

    def play_step(self, action: int) -> tuple[float, bool, int]:
        """Apply action (0 straight, 1 right, 2 left), then return reward/done/score."""
        if self.show_window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                    return 0.0, True, self.score

        self.frame_iteration += 1
        if action == 1:
            self.direction_index = (self.direction_index + 1) % 4
        elif action == 2:
            self.direction_index = (self.direction_index - 1) % 4

        dx, dy = self.DIRECTIONS[self.direction_index]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # A move into the current tail cell is legal if the tail moves away.
        ate_food = head == self.food
        collision_body = self.snake if ate_food else self.snake[:-1]
        x, y = head
        if x < 0 or x >= self.columns or y < 0 or y >= self.rows or head in collision_body:
            self._draw(final=True)
            return -10.0, True, self.score

        self.snake.insert(0, head)
        reward = -0.1
        if ate_food:
            self.score += 1
            reward = 10.0
            self.food = self._place_food()
        else:
            self.snake.pop()

        # This prevents an untrained agent from looping forever in one game.
        if self.frame_iteration > 100 * len(self.snake):
            self._draw(final=True)
            return -10.0, True, self.score

        self._draw()
        return reward, False, self.score

    def _draw(self, final: bool = False) -> None:
        if not self.show_window or self.screen is None:
            return
        self.screen.fill(self.BLACK)
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRID, (0, y), (self.width, y))

        for index, (x, y) in enumerate(self.snake):
            color = self.GREEN if index == 0 else self.DARK_GREEN
            rect = pygame.Rect(x * self.cell_size + 1, y * self.cell_size + 1,
                               self.cell_size - 2, self.cell_size - 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

        fx, fy = self.food
        food_rect = pygame.Rect(fx * self.cell_size + 3, fy * self.cell_size + 3,
                                self.cell_size - 6, self.cell_size - 6)
        pygame.draw.ellipse(self.screen, self.RED, food_rect)
        label = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(label, (8, 6))
        if final:
            message = self.font.render("Game over", True, self.WHITE)
            self.screen.blit(message, (self.width // 2 - message.get_width() // 2, 8))
        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self) -> None:
        pygame.quit()

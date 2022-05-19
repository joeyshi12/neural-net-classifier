from typing import List
from pygame import Surface
from pygame.draw import rect


class Canvas:
    canvas: List[List[int]]
    square_length: int

    def __init__(self, width: int, height: int, square_length: int):
        if width <= 0 or height <= 0:
            raise Exception("Invalid canvas width and height")
        self.canvas = [[0] * width for _ in range(height)]
        self.square_length = square_length

    def fill_square(self, row: int, col: int) -> None:
        if 0 <= row < len(self.canvas) and 0 <= col < len(self.canvas[0]):
            self.canvas[row][col] = 1

    def draw(self, surface: Surface) -> None:
        width, height = len(self.canvas), len(self.canvas[0])
        for i in range(height):
            for j in range(width):
                if self.canvas[i][j] == 1:
                    rect(surface, (0, 0, 0), (10 * j, 10 * i, self.square_length, self.square_length))

    def clear(self) -> None:
        for row in self.canvas:
            for i, _ in enumerate(row):
                row[i] = 0

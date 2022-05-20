import sys
import joblib
from typing import List
import numpy as np
import pygame as pg


def paint_loop(model_file: str) -> None:
    pg.init()
    pg.display.set_caption("Digit Recognition")
    clock = pg.time.Clock()
    surface = pg.display.set_mode((560, 560))
    canvas = Canvas(56, 56, 10)
    model = joblib.load(model_file)
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(1)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    yhat = model.predict([compress(canvas.canvas).flatten()])[0]
                    print(yhat)
                    canvas.clear()
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                row, col = y // canvas.square_length, x // canvas.square_length
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        canvas.fill_square(row + i, col + j)

        surface.fill((255, 255, 255))
        canvas.draw(surface)
        pg.display.flip()
        clock.tick(0)


def compress(canvas):
    compressed_canvas = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            compressed_canvas[i][j] = (canvas[2 * i][2 * j]
                                       + canvas[2 * i + 1][2 * j]
                                       + canvas[2 * i][2 * j + 1]
                                       + canvas[2 * i + 1][2 * j + 1]) / 4
    return compressed_canvas


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

    def draw(self, surface: pg.Surface) -> None:
        width, height = len(self.canvas), len(self.canvas[0])
        for i in range(height):
            for j in range(width):
                if self.canvas[i][j] == 1:
                    pg.draw.rect(surface, (0, 0, 0), (10 * j, 10 * i, self.square_length, self.square_length))

    def clear(self) -> None:
        for row in self.canvas:
            for i, _ in enumerate(row):
                row[i] = 0

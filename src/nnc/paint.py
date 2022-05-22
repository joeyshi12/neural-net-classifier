import sys
import joblib
from typing import List
import numpy as np
import pygame as pg

from nnc.nnclassifier import NNClassifier


class PaintManager:
    radius: int = 24
    unit_length: int = 4

    def __init__(self, surface: pg.Surface, clock: pg.time.Clock, model: NNClassifier, heading_size: int):
        pg.init()
        pg.display.set_caption("Digit Recognition")
        self.font = pg.font.SysFont("lucidaconsole", 16, bold=True)
        self.surface = surface
        self.clock = clock
        self.model = model
        self.heading_size = heading_size

        width, height = surface.get_size()
        surface.fill((255, 255, 255))  # fill surface with white pixels
        pg.draw.rect(surface, (160, 160, 160), (0, 0, width, heading_size))  # fill heading area with grey pixels
        self.canvas = np.zeros((width // self.unit_length, width // self.unit_length))

    def handle_event(self, event: pg.event.Event) -> None:
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit(1)
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                self.predict()
            elif event.key == pg.K_c:
                self.clear()

    def update(self):
        if pg.mouse.get_pressed()[0]:
            x, y = pg.mouse.get_pos()
            row = (y - self.heading_size) // self.unit_length
            col = x // self.unit_length
            self.draw(row, col)
        pg.display.flip()
        self.clock.tick(0)

    def draw(self, row: int, col: int):
        rows, cols = self.canvas.shape
        for i in range(-self.radius, self.radius + 1):
            for j in range(-self.radius, self.radius + 1):
                if 0 > row + i or row + i >= rows or 0 > col + j or col + j >= cols:
                    continue
                if i ** 2 + j ** 2 > self.radius:
                    continue
                if not self.canvas[row + i][col + j]:
                    self.canvas[row + i][col + j] = 1
                    rect = (
                        self.unit_length * (col + j),
                        self.heading_size + self.unit_length * (row + i),
                        self.unit_length,
                        self.unit_length
                    )
                    pg.draw.rect(self.surface, (0, 0, 0), rect)

    def predict(self):
        data = np.zeros((28, 28))
        cell_length = self.canvas.shape[0] // 28
        for i in range(28):
            for j in range(28):
                data[i][j] = np.average(
                    self.canvas[cell_length * i:cell_length * (i + 1), cell_length * j:cell_length * (j + 1)]
                )
        digit = self.model.predict([data.flatten()])[0]
        width, _ = self.surface.get_size()
        pg.draw.rect(self.surface, (160, 160, 160), (0, 0, width, self.heading_size))
        prediction_message = self.font.render(f"Digit: {digit}", True, (0, 0, 0))
        self.surface.blit(prediction_message, (10, 10))

    def clear(self):
        for row in self.canvas:
            for i, _ in enumerate(row):
                row[i] = 0
        width, height = self.surface.get_size()
        rect = (0, self.heading_size, width, height - self.heading_size)
        pg.draw.rect(self.surface, (255, 255, 255), rect)


def paint_loop(model_file: str) -> None:
    width, padding_top = 560, 40
    manager = PaintManager(
        surface=pg.display.set_mode((width, width + padding_top)),
        clock=pg.time.Clock(),
        model=joblib.load(model_file),
        heading_size=padding_top
    )
    while True:
        for event in pg.event.get():
            manager.handle_event(event)
        manager.update()

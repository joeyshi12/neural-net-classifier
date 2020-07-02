import sys
import pygame
from pygame.locals import *
import numpy as np
from joblib import load
from tkinter import *
from tkinter import messagebox
Tk().wm_withdraw()
pygame.init()
pygame.display.set_caption('Digit Recognition')

fps = 160
fpsClock = pygame.time.Clock()
surface = pygame.display.set_mode((560, 560))
canvas = np.zeros((56, 56))
block_length = 10
model = load('data/mlp.joblib')


def fillSquare(canvas, i, j):
    n, d = canvas.shape
    if 0 <= i < n and 0 <= j < d:
        canvas[i][j] = 1


def draw_canvas(surface, canvas):
    n, d = canvas.shape
    for i in range(n):
        for j in range(d):
            if canvas[i][j] == 1:
                pygame.draw.rect(surface, (0,0,0), (10 * j, 10 * i, block_length, block_length))


def compress(canvas):
    retVal = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            retVal[i][j] = (canvas[2 * i][2 * j] + canvas[2 * i + 1][2 * j] + canvas[2 * i][2 * j + 1] + canvas[2 * i + 1][2 * j + 1]) / 4
    return retVal


if __name__ == '__main__':
    while True:
      for event in pygame.event.get():
        if event.type == QUIT:
          pygame.quit()
          sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                y_hat = model.predict([compress(canvas).flatten()])[0]
                messagebox.showinfo('Prediction', y_hat)
                canvas.fill(0)

      if pygame.mouse.get_pressed()[0]:
          i = int(pygame.mouse.get_pos()[1] / block_length)
          j = int(pygame.mouse.get_pos()[0] / block_length)
          for y in [i - 1, i, i + 1]:
              for x in [j - 1, j, j + 1]:
                  fillSquare(canvas, y, x)

      surface.fill((255,255,255))
      draw_canvas(surface, canvas)

      pygame.display.flip()
      fpsClock.tick(fps)

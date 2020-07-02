import sys
import pygame
from pygame.locals import *
import numpy as np
from joblib import load
from tkinter import *
from tkinter import messagebox
Tk().wm_withdraw() #to hide the main window

pygame.init()

fps = 60
fpsClock = pygame.time.Clock()
length = 560
screen = pygame.display.set_mode((length, length))

canvas_length = int(length / 10)
block_length = int(length / canvas_length)
blocks = []


def fillSquare(canvas, i, j):
    if 0 <= i < len(canvas) and 0 <= j < len(canvas[0]):
        canvas[i][j] = 1
    block = pygame.Rect((block_length * j, block_length * i, block_length, block_length))
    blocks.append(block)


def compress(canvas):
    retVal = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            retVal[i][j] = (canvas[2*i][2*j] + canvas[2*i+1][2*j] + canvas[2*i][2*j+1] + canvas[2*i+1][2*j+1]) / 4
    return retVal


def main():
    model = load('model.joblib')
    canvas = np.zeros((canvas_length, canvas_length))
    while True:
      for event in pygame.event.get():
        if event.type == QUIT:
          pygame.quit()
          sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                y_hat = model.predict([compress(canvas).flatten()])[0]
                messagebox.showinfo('Prediction',y_hat)
                canvas = np.zeros((canvas_length, canvas_length))
                blocks.clear()

      if pygame.mouse.get_pressed()[0]:
          i = int(pygame.mouse.get_pos()[1] / block_length)
          j = int(pygame.mouse.get_pos()[0] / block_length)
          for y in [i - 1, i, i + 1]:
              for x in [j - 1, j, j + 1]:
                  fillSquare(canvas, y, x)

      screen.fill((255,255,255))
      for block in blocks:
          pygame.draw.rect(screen, (0,0,0), block)

      pygame.display.flip()
      fpsClock.tick(fps)


if __name__ == '__main__':
    main()

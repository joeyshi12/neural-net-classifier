import sys
import argparse
import gzip
import pickle
import logging
from typing import List
import pygame as pg
import numpy as np
from neural_net import NNClassifier
from canvas import Canvas


def train_model(infile: str, outfile: str) -> None:
    with gzip.open(infile) as file:
        train_set, valid_set, test_set = pickle.load(file, encoding="latin1")

    X, y = train_set
    X_test, y_test = test_set
    n, d = X.shape
    model = NNClassifier()
    model.fit(X, y)
    logging.info("n = %d, d = %d", n, d)

    # Compute training error
    yhat = model.predict(X)
    train_error = np.mean(yhat != y)
    logging.info("Training error = %f", train_error)

    # Compute test error
    yhat = model.predict(X_test)
    test_error = np.mean(yhat != y_test)
    logging.info("Test error = %f", test_error)

    # Save weights
    model.save(outfile)


def compress(canvas):
    retVal = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            retVal[i][j] = (
                canvas[2 * i][2 * j] \
                + canvas[2 * i + 1][2 * j] \
                + canvas[2 * i][2 * j + 1] \
                + canvas[2 * i + 1][2 * j + 1]) / 4
    return retVal


def paint_loop(model_file: str) -> None:
    pg.init()
    pg.display.set_caption('Digit Recognition')
    clock = pg.time.Clock()
    surface = pg.display.set_mode((560, 560))
    canvas = Canvas(56, 56, 10)
    model = NNClassifier()
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit(1)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    yhat = model.predict([compress(canvas).flatten()])[0]
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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("infile", help="File path to zip training data")
    train_parser.add_argument("outfile", help="Path to output weights", default="model.npy")

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("model_file", help="File path to model weights")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.infile, args.outfile)
    else:
        paint_loop(args.model_file)


if __name__ == '__main__':
    main()

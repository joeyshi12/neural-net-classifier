import argparse
import gzip
import pickle
import joblib
import numpy as np
from nnclassifier.nnclassifier import NNClassifier


def train_model(infile: str, outfile: str) -> None:
    with gzip.open(infile) as file:
        train_set, valid_set, test_set = pickle.load(file, encoding="latin1")

    X, y = train_set
    X_test, y_test = test_set
    n, d = X.shape
    print(f"n = {n}, d = {d}")
    model = NNClassifier()
    model.fit(X, y)

    # Compute training error
    yhat = model.predict(X)
    train_error = np.mean(yhat != y)
    print(f"Training error = {train_error}")

    # Compute test error
    yhat = model.predict(X_test)
    test_error = np.mean(yhat != y_test)
    print(f"Test error = {test_error}")

    # Save model
    joblib.dump(model, outfile)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("infile", help="File path to zip training data")
    train_parser.add_argument("outfile", help="Path to output weights", default="model.npy")

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("model_file", help="File path to model weights")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.infile, args.outfile)
    else:
        from nnclassifier.paint import paint_loop
        paint_loop(args.model_file)


if __name__ == '__main__':
    main()

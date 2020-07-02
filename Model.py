import os
import pickle
import gzip
import argparse
import numpy as np
from joblib import dump
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set
    print("n =", X.shape[0])
    print("d =", X.shape[1])

    if question == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(300,300), verbose=True)
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

        dump(model, 'data/mlp.joblib')

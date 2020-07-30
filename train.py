import pickle
import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model.neural_net import NNClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    X_test, y_test = test_set
    print("n =", X.shape[0])
    print("d =", X.shape[1])

    if question == 'train':
        model = NNClassifier(hidden_layer_sizes=[300], lammy=0.01, epochs=20, verbose=True)
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(X_test)
        testError = np.mean(yhat != y_test)
        print("Test error     = ", testError)

    elif question == 'kmeans':
        model = KMeans(n_clusters=10, verbose=1)
        model.fit(X, y)

        fig, ax = plt.subplots(2, 5)
        for i in range(2):
            for j in range(5):
                im = model.cluster_centers_[i * 5 + j]
                ax[i, j].set_title('Digit: %d' % (model.predict([im])[0]))
                ax[i, j].imshow(1 - im.reshape((28, 28)), cmap='gray')
        fig.tight_layout()
        plt.savefig('figs/kmeans.png')

    elif question == 'pca':
        model = PCA(2)
        Z = model.fit_transform(X)
        n = 100

        plt.figure()
        plt.scatter(Z[:n, 0], Z[:n, 1], c=y[:n])
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title('PCA')

        plt.savefig('figs/pca.png')

    else:
        print("Unknown question: %s" % question)

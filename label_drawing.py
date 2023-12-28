import matplotlib.pyplot as plt
import numpy as np
from label_spreading import predict
def label_drawing(labels, k_centers, classes, label_distributions, X, sigma, delta, label, method):
    y = predict(classes, label_distributions, X, sigma, delta, label, method)
    targets = list(labels.keys())
    plt.figure(figsize=(10, 4))
    for i in range(len(labels)):
        plt.scatter(X[:, 0][y == targets[i]], X[:, 1][y == targets[i]], label=targets[i])
    plt.scatter(X[:, 0][k_centers], X[:, 1][k_centers], c='k', label='centers')
    plt.legend()
    plt.show()

def draw_decision(datas, rhos, deltas):
    n = np.shape(datas)[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
        plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
        plt.xlabel('local density-ρ')
        plt.ylabel('minimum distance to higher density points-δ')
    plt.show()

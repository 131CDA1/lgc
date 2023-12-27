from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
def gaussian_kernel(X, sigma=20):
    K = euclidean_distances(X,squared=True)
    K *= -sigma
    np.exp(K, K)
    return K
def density_based_kernel(X, sigma, delta, label):
    n = np.shape(X)[0]

    K = (euclidean_distances(X) - delta) ** 2
    K *= -sigma
    # K -= np.max(K)  # 防止数据溢出
    np.exp(K, K)

    K = (K.T + K) * 0.5
    K = K.flatten()
    K[label] = 0
    K = K.reshape(n, n)
    np.fill_diagonal(K, 0)
    return K
def get_kernel(X, sigma, delta, label, type=""):
    if type == 'knn':
        return gaussian_kernel(X, sigma)
    elif type == 'dng':
        return density_based_kernel(X, sigma, delta, label)
    else:
        return "type变量名错误"

import numpy as np
from scipy.sparse import csgraph
from kernel import get_kernel
from sklearn import metrics


def build_graph(X, sigma, delta, label, method):
    # 计算标准化后的拉普拉斯矩阵
    n_samples = X.shape[0]
    # 切换knn或dng
    affinity_matrix = get_kernel(X, sigma, delta, label, method)
    laplacian = -csgraph.laplacian(affinity_matrix, normed=True)
    laplacian.flat[::n_samples + 1] = 0.0  # 设置对角线原始全为0
    return laplacian


def fit(X, y, sigma, delta, label, method, alpha=0.2):
    """
    模型拟合
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A matrix of shape (n_samples, n_samples) will be created from this.

    y : array-like of shape (n_samples,)
        `n_labeled_samples` (unlabeled points are marked as -1)
        All unlabeled samples will be transductively assigned labels.
    """
    # self.X_ = X
    graph_matrix = build_graph(X, sigma, delta, label, method)
    classes = np.unique(y)
    classes = (classes[classes != -1])

    n_samples, n_classes = len(y), len(classes)

    # alpha = self.alpha
    if alpha is None or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha必须大于0小于1")
    label_distributions = np.zeros((n_samples, n_classes))
    for label in classes:
        label_distributions[y == label, classes == label] = 1

    # 非迭代法
    inv = np.linalg.inv(np.eye(graph_matrix.shape[0]) - alpha * graph_matrix)
    label_distributions = np.matmul(inv, label_distributions)

    normalizer = np.sum(label_distributions, axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1
    label_distributions /= normalizer
    # transduction = classes[np.argmax(label_distributions, axis=1)]
    return classes, label_distributions


def predict(classes, label_distributions, X, sigma, delta, label, method):
    """
    预测
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix.
    Returns
    -------
    y : ndarray of shape (n_samples,)
        Predictions for input data.
    """
    probas = predict_proba(label_distributions, X, sigma, delta, label, method)
    return classes[np.argmax(probas, axis=1)].ravel()


def predict_proba(label_distributions, X, sigma, delta, label, method):
    """
    概率预测
    Parameters
    ----------
    Returns
    -------
    probabilities : shape (n_samples, n_classes)
    """
    # 切换knn或dng
    weight_matrices = get_kernel(X, sigma, delta, label, method)

    weight_matrices = weight_matrices.T
    probabilities = np.matmul(weight_matrices, label_distributions)
    normalizer = np.sum(probabilities, axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1  # 平滑处理
    probabilities /= normalizer
    return probabilities


def score(classes, y, label_distributions, X, sigma, delta, label, method):
    acc = metrics.accuracy_score(predict(classes, label_distributions, X, sigma, delta, label, method), y)
    return metrics.classification_report(predict(classes, label_distributions, X, sigma, delta, label, method), y,
                                         zero_division=1), acc

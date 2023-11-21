from scipy.sparse import csgraph
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.datasets import load_iris
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pandas as pd
def load_data(data_raw,true_labels,num:int= 1,random_seed=None):
    random.seed(random_seed)
    labels = {}
    for index, value in enumerate(true_labels):
        if value not in labels.keys():
            labels[value] = [index]
        else:
            labels.get(value).append(index)
    for value in labels.values():
        random.shuffle(value)
    label = []
    for i in range(num):
        for value in labels.values():
            label.append(value[i])
    # print(label)
    # print(random_unlabeled_points)
    mixed_labels = deepcopy(true_labels)
    for i in range(len(mixed_labels)):
        # print(i)
        if i not in label:
            mixed_labels[i] = -1
    # for i in range(len(y_mixed)):
    #     if y_mixed[i] == -1:
    #         continue
    #     if rng.random() < noise_rate:  # 将有标签的样本中的noise_rate替换为随机标签
    #         candidate_ids = np.random.permutation(n_class).tolist()
    #         candidate_ids.remove(y_mixed[i])
    #         y_mixed[i] = candidate_ids[0]
    ss = StandardScaler()
    data_raw = ss.fit_transform(data_raw)
    return data_raw, true_labels, mixed_labels, labels

def kernel(X, y=None, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, y, squared=True)
    K *= -gamma
    np.exp(K, K)  # <==> K = np.exp(K)
    return K


class LabelSpreading():
    """Base class for label propagation module.
    """

    def __init__(self, gamma=20, alpha=0.2):
        self.gamma = gamma
        self.alpha = alpha

    def _get_kernel(self, X, y=None):
        return kernel(X, y, gamma=self.gamma)

    def _build_graph(self):
        # 计算标准化后的拉普拉斯矩阵
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        # D^{-1/2}WD^{-1/2}
        laplacian = -csgraph.laplacian(affinity_matrix, normed=True)
        laplacian.flat[::n_samples + 1] = 0.0  # 设置对角线原始全为0
        return laplacian

    def fit(self, X, y):
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
        self.X_ = X
        self.graph_matrix = self._build_graph()
        classes = np.unique(y)
        self.classes_ = (classes[classes != -1])

        n_samples, n_classes = len(y), len(self.classes_)

        alpha = self.alpha
        if alpha is None or alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha必须大于0小于1")
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in self.classes_:
            self.label_distributions_[y == label, self.classes_ == label] = 1

        # 非迭代法
        inv = np.linalg.inv(np.eye(self.graph_matrix.shape[0]) - self.alpha * self.graph_matrix)
        self.label_distributions_ = np.matmul(inv, self.label_distributions_)

        normalizer = np.sum(self.label_distributions_, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer
        self.transduction_ = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        return self

    def predict(self, X):
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
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    def predict_proba(self, X):
        """
        概率预测
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        Returns
        -------
        probabilities : shape (n_samples, n_classes)
        """
        weight_matrices = self._get_kernel(self.X_, X)
        weight_matrices = weight_matrices.T
        probabilities = np.matmul(weight_matrices, self.label_distributions_)
        normalizer = np.sum(probabilities, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1  # 平滑处理
        probabilities /= normalizer
        return probabilities

    def score(self, X, y):
        # print(self.predict(X),y)
        acc = metrics.accuracy_score(self.predict(X), y)
        return metrics.classification_report(self.predict(X), y, zero_division=1),acc

    def label_drawing(self, X, data_raw,labels):
        y = self.predict(X)
        # iris = load_iris()
        X = data_raw
        targets = list(labels.keys())
        # features = iris.feature_names
        plt.figure(figsize=(10, 4))
        for i in range(len(labels)):
            plt.scatter(X[:, 0][y == i], X[:, 1][y == i],label=targets[i])
        # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs', label=targets[0])
        # plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'kx', label=targets[1])
        # plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'ro', label=targets[2])
        # plt.xlabel(features[0])
        # plt.ylabel(features[1])
        plt.legend()
        plt.show()


def test_label_spreading():
    # data_raw, true_labels = load_iris(return_X_y=True)
    data_raw,true_labels = load_titanic('train.csv')
    x, y, y_mixed,labels = load_data(data_raw,true_labels,num=1,random_seed=2023)
    model = LabelSpreading()
    model.fit(x, y_mixed)
    score, acc = model.score(x, y)
    print(score)
    print('准确度为',acc)
    model.label_drawing(x,data_raw,labels)
    # print(len(x),len(y))


def load_titanic(path):
    data = pd.read_csv(path, header=0)
    data = data.drop(columns=['PassengerId', 'Name', 'Sex' ,'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    # data.drop(axis=0)
    survived = data['Survived']
    survived = survived.to_numpy()
    data = data.drop('Survived', axis=1)
    data = data.to_numpy()
    return data, survived
if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    test_label_spreading()

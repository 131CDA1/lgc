# import logging
from scipy.sparse import csgraph
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
def load_data(noise_rate=0.1):
    n_class = 10
    x, y = load_iris(return_X_y=True)
    # 划分30%为测试集，70%为训练集
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3)
    rng = np.random.RandomState(20)
    # 在训练集中将其中80%样本的标签去掉，置为-1
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.8
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    for i in range(len(y_mixed)):
        if y_mixed[i] == -1:
            continue
        if rng.random() < noise_rate:  # 在训练集中，将有标签的样本中的noise_rate替换为随机标签
            candidate_ids = np.random.permutation(n_class).tolist()
            candidate_ids.remove(y_mixed[i])
            y_mixed[i] = candidate_ids[0]
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test, y_mixed

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
        # return accuracy_score(self.predict(X), y)
        return classification_report(self.predict(X), y, zero_division=1)

def test_label_spreading():
    x_train, x_test, y_train, y_test, y_mixed = load_data(noise_rate=0.1)
    model = LabelSpreading()
    model.fit(x_train, y_mixed)

    print("训练集准确率:\n",model.score(x_train, y_train))
    print("测试集准确率:\n",model.score(x_test, y_test))
    # print(len(x_train),len(y_train))
    # logging.info({classification_report(x_train,y_train)})
if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    # logging.basicConfig(level=logging.DEBUG,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
    #                     format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
    #                     datefmt='%Y-%m-%d %H:%M:%S', )
    test_label_spreading()


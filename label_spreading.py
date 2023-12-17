from scipy.sparse import csgraph
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pandas as pd
def load_data_bynum(data_raw,true_labels,num= 1,random_seed=None):
    """
    随机抽取样本点，其保证在random_seed相同时，num=2是在num=1的情况下增加了1个有标签的样本
    :param data_raw: 原数据集
    :param true_labels: 原数据集的标签
    :param num: 每个聚类中选取的样本点个数
    :param random_seed: 随机数种子
    :return:mixed_label表示抽取后的样本标签,labels表示各个聚类的抽签顺序
    """
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
    k = len(labels.values())
    # print(label)
    # print(random_unlabeled_points)
    mixed_labels = deepcopy(true_labels)
    for i in range(len(mixed_labels)):
        # print(i)
        if i not in label:
            mixed_labels[i] = -1
    ss = StandardScaler()
    data_raw = ss.fit_transform(data_raw)
    return data_raw, true_labels, mixed_labels, labels, label, k

def load_data_bypersent(data_raw,true_labels,percent=0.02,random_seed=None):
    num = int(len(data_raw)*percent)
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
    k = len(labels.values())
    # print(label)
    # print(random_unlabeled_points)
    mixed_labels = deepcopy(true_labels)
    for i in range(len(mixed_labels)):
        # print(i)
        if i not in label:
            mixed_labels[i] = -1
    ss = StandardScaler()
    data_raw = ss.fit_transform(data_raw)
    return data_raw, true_labels, mixed_labels, labels, label, k

def get_distance_matrix(datas):
    n = np.shape(datas)[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v_i = datas[i, :]
            v_j = datas[j, :]
            distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
    return distance_matrix

def density_based_kernel(X, sigma, label):
    n = np.shape(X)[0]

    delta = get_delta(X,label)
    delta = np.tile(delta,np.shape(delta)[0])
    delta = delta.reshape(n,n)

    K = euclidean_distances(X) - delta
    K *= -sigma
    # K -= np.max(K)  # 防止数据溢出
    np.exp(K, K)

    K =(K.T + K)* 0.5
    K = K.flatten()
    K[label] = 0
    K = K.reshape(n,n)
    np.fill_diagonal(K,0)
    return K

def gaussian_kernel(X,sigma= 20):
    K = euclidean_distances(X)
    K *= -sigma
    np.exp(K, K)  # <==> K = np.exp(K)
    return K
def select_dc(distance_matrix,percent=0.02):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)     # 将300x300的距离矩阵铺平为90000x1的向量
    position = int(n * (n - 1) * percent)
    dc = np.sort(distance_array)[position + n]
    # 取数据集的第2%的距离当做dc
    return dc
def get_local_density(distance_matrix, dc, method=None):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        if method is None:
            rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
        else:
            pass
    # 直接对每个点周围距离小于dc的点进行计数,输出一个300的密度向量
    return rhos
def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)
    nearest_neighbor = np.zeros(n)
    rhos_index = np.argsort(-rhos)  # 得到密度ρ从大到小的排序的索引
    for i, index in enumerate(rhos_index):
        if i == 0:
            continue
        higher_rhos_index = rhos_index[:i]
        deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
        nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])
        nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
    deltas[rhos_index[0]] = np.max(deltas)
    return deltas, nearest_neighbor
def draw_decision(datas, rhos, deltas):
    n = np.shape(datas)[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
        plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
        plt.xlabel('local density-ρ')
        plt.ylabel('minimum distance to higher density points-δ')
    plt.show()

def find_k_centers(distance_matrix, percent, k):
    dc = select_dc(distance_matrix,percent)
    rhos = get_local_density(distance_matrix, dc)
    deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    draw_decision(distance_matrix,rhos,deltas)
    return centers[:k]

def get_delta(X,label):
    n = X.shape[0]
    d = euclidean_distances(X)
    delta = np.zeros(n)
    for i in range(n):
        if i in label:
            delta[i] = min(d[i])
        else:
            delta[i] = max(d[i])
    return delta


class LabelSpreading():
    """Base class for label propagation module.
    """

    def __init__(self, type, alpha=0.2, b=0.5):
        self.alpha = alpha
        self.b = b
        self.type = type
    def _get_kernel(self, X, sigma, label, type = ""):
        if type == 'knn':
            return gaussian_kernel(X,sigma)
        elif type == 'dng':
            return density_based_kernel(X, sigma, label)
        else:
            return "type变量名错误"
    def _get_constant(self, X, label):
        num = 1 / len(label)
        k = X.copy()
        k = k[label]
        data = np.dot(X, k.T)
        data = np.sum(data, axis=1)
        sigma = self.b * num * data
        self.sigma = sigma

        n = X.shape[0]
        d = euclidean_distances(X)
        delta = np.zeros(n)
        for i in range(n):
            if i in label:
                delta[i] = min(d[i])
            else:
                delta[i] = max(d[i])
        self.delta = delta
        self.label = label
        return sigma,delta,label

    def _build_graph(self):
        # 计算标准化后的拉普拉斯矩阵
        n_samples = self.X_.shape[0]
        # 切换knn或dng
        affinity_matrix = self._get_kernel(self.X_, self.sigma, self.label, self.type)
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
        probas = self.predict_proba()
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    def predict_proba(self):
        """
        概率预测
        Parameters
        ----------
        Returns
        -------
        probabilities : shape (n_samples, n_classes)
        """
        #切换knn或dng
        weight_matrices = self._get_kernel(self.X_, self.sigma, self.label, self.type)

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

    def label_drawing(self, X, labels, k_centers):
        y = self.predict(X)
        # iris = load_iris()
        targets = list(labels.keys())
        # features = iris.feature_names
        plt.figure(figsize=(10, 4))
        # plt.figure()
        # print(X,y,labels,targets)
        for i in range(len(labels)):
            plt.scatter(X[:, 0][y == targets[i]], X[:, 1][y == targets[i]],label=targets[i])
        plt.scatter(X[:, 0][k_centers], X[:, 1][k_centers], c='k', label='centers')
        # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs', label=targets[0])
        # plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'kx', label=targets[1])
        # plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'ro', label=targets[2])
        # plt.xlabel(features[0])
        # plt.ylabel(features[1])
        plt.legend()
        plt.show()
def load_excel(path):
    full_data = pd.read_excel(path)
    true_labels = full_data.iloc[:,-1]
    data = full_data.iloc[:,0:-1]
    data = data.to_numpy()
    true_labels = true_labels.to_numpy()
    # print(data,true_labels)
    data_zscore = (data - data.mean(axis=0))/data.std(axis=0)
    data_maxmin = (data - data.min())/(data.max() - data.min())

    # print(data_maxmin)
    return data, true_labels
def test_label_spreading():
    # 加载文件位置
    # data_raw, true_labels = load_iris(return_X_y=True)
    data_raw, true_labels = load_excel('real_raw/R15.xlsx')
    # 选取样本个数或总百分比
    # x, y, y_mixed, labels, label, k = load_data_bynum(data_raw,true_labels,random_seed=2023,num=2)
    x, y, y_mixed, labels, label, k = load_data_bypersent(data_raw,true_labels,random_seed=None,percent=0.02)
    alpha = 0.2
    b = 0.5
    k_centers = find_k_centers(euclidean_distances(data_raw), 0.02, k)
    # model_knn = LabelSpreading(type='knn', alpha=alpha, b=b)
    # model_knn._get_constant(data_raw, euclidean(get_distance_matrix(data_raw), 0.02, k))
    # model_knn.fit(x, y_mixed)

    model_dng = LabelSpreading(type='dng', alpha=alpha, b=b)
    model_dng._get_constant(data_raw, k_centers)
    model_dng.fit(x, y_mixed)

    # score_knn, acc_knn = model_knn.score(x, y)
    # print('knn算法:')
    # print(score_knn)
    # print('准确度为', acc_knn)
    # model_knn.label_drawing(x, labels)

    score_dng , acc_dng = model_dng.score(x, y)
    print('dng算法:')
    print(score_dng)
    print('准确度为', acc_dng)
    model_dng.label_drawing(x, labels, k_centers)
if __name__ == '__main__':
    test_label_spreading()

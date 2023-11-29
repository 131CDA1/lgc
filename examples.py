import numpy as np
import matplotlib.pyplot as plt


def get_distance_matrix(datas):
    n = np.shape(datas)[0]
    distance_matrix = np.zeros((n, n))
    # 生成一个300x300的全零矩阵
    for i in range(n):
        for j in range(n):
            v_i = datas[i, :]
            v_j = datas[j, :]
            # v_i和v_j将datas里的数据分别拿出来
            distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
            # 这个distance_matrix的结构是这样的
            # 横纵坐标就是对应的两点，格子里是值
            # distance  1  2  3  4
            #    1     11 12 13 14
            #    2     21 22 23 24
            #    3     31 32 33 34
    return distance_matrix


def select_dc(distance_matrix):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)     # 将300x300的距离矩阵铺平为90000x1的向量
    percent = 2.0 / 100
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


def find_k_centers(rhos, deltas, k):
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]


def density_peal_cluster(rhos, centers, nearest_neighbor):
    k = np.shape(centers)[0]
    if k == 0:
        print("Can't find any center")
        return
    n = np.shape(rhos)[0]
    labels = -1 * np.ones(n).astype(int)

    for i, center in enumerate(centers):
        labels[center] = i

    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if labels[index] == -1:
            labels[index] = labels[int(nearest_neighbor[index])]
    return labels


def generate_gauss_datas():
    first_group = np.random.normal(20, 1.2, (100, 2))
    second_group = np.random.normal(10, 1.2, (100, 2))
    third_group = np.random.normal(15, 1.2, (100, 2))
    # numpy.random.normal(loc:均值, scale:标准差, size:大小) 从正态分布中随机取值
    # 生成了三个聚类 大小为各100的二维数据

    datas = []
    for i in range(100):
        datas.append(first_group[i])
        datas.append(second_group[i])
        datas.append(third_group[i])
    datas = np.array(datas)
    # 将三个聚类按顺序放进datas里
    return datas


def draw_decision(datas, rhos, deltas):
    n = np.shape(datas)[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
        plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
        plt.xlabel('local density-ρ')
        plt.ylabel('minimum distance to higher density points-δ')
    plt.show()


def main():
    datas = generate_gauss_datas()
    distance_matrix = get_distance_matrix(datas)
    dc = select_dc(distance_matrix)
    rhos = get_local_density(distance_matrix, dc)
    deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)
    centers = find_k_centers(rhos, deltas, 3)
    labels = density_peal_cluster(rhos, centers, nearest_neighbor)
    draw_decision(datas, rhos, deltas)
    plt.cla()
    fig, ax = plt.subplots()
    for i in range(300):
        if labels[i] == 0:
            ax.scatter(datas[i, 0], datas[i, 1], facecolor='C0', edgecolors='k')
        elif labels[i] == 1:
            ax.scatter(datas[i, 0], datas[i, 1], facecolor='yellow', edgecolors='k')
        elif labels[i] == 2:
            ax.scatter(datas[i, 0], datas[i, 1], facecolor='red', edgecolors='k')
    plt.show()


if __name__ == '__main__':
    main()

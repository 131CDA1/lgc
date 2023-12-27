import numpy as np
def select_dc(distance_matrix, percent=0.02):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)
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
    return rhos
def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)
    nearest_neighbor = np.zeros(n)
    rhos_index = np.argsort(-rhos)
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
def get_sigma(data_raw, label, b=0.5):
    num = 1 / len(label)
    k = data_raw.copy()
    k = k[label]
    data = np.dot(data_raw, k.T)
    data = np.sum(data, axis=1)
    sigma = b * num * data
    return sigma
from label_spreading import *
from examples import *
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
def find_k_centers(distance_matrix, percent, k):
    dc = select_dc(distance_matrix,percent)
    rhos = get_local_density(distance_matrix, dc)
    deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]

if __name__ == '__main__' :
    data_raw, true_labels = load_excel('./real_raw/iris.xlsx')
    distance = get_distance_matrix(data_raw)
    dc1 = select_dc(distance)
    rhos1 = get_local_density(distance, dc1)
    delta1, nearest_neighbor = get_deltas(distance, rhos1)
    # matrix = rhos1 * delta1
    # print(matrix)
    center1 = find_k_centers(distance, 0.02, 3)
    draw_decision(data_raw, rhos1, delta1)
    # kernel = gaussian_kernel(data_raw)
    # dc2 = select_dc(kernel)
    # rhos2 = get_local_density(kernel, dc2)
    # delta2, nearest_neighbor = get_deltas(kernel, rhos2)
    # center2 = find_k_centers(kernel,0.02, 3)
    # draw_decision(data_raw, rhos2, delta2)
    print(center1)
    # print(center2)

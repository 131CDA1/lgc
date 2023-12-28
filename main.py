from load_data import *
from find_centers import *
from label_spreading import *
from sklearn.metrics.pairwise import euclidean_distances
from label_drawing import *

if __name__ == '__main__':
    data_raw, true_labels = load_excel('dataset/balance_scale.xlsx')
    x, y, y_mixed, labels, label, k = load_data_bypersent(data_raw, true_labels, random_seed=None, percent=0.02)
    distance = euclidean_distances(x)
    dc = select_dc(distance, percent=0.02)
    rhos = get_local_density(distance, dc)
    delta, nearest_neighbor = get_deltas(distance, rhos)
    rhos_centers = np.argsort(-rhos)[:k]
    centers = find_k_centers(rhos, delta, k)
    sigma = get_sigma(data_raw, rhos_centers, b=0.5)
    classes, label_distributions = fit(x, y_mixed, sigma, delta, centers, method='dng', alpha=0.2)
    score, acc = score(classes, y, label_distributions, data_raw, sigma, delta, label, method='dng')
    print(score)
    print('准确度为', acc)
    draw_decision(data_raw, rhos, delta)
    label_drawing(labels, centers, classes, label_distributions, x, sigma, delta, label, method='dng')

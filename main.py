import random
random.seed(10)
true_labels = [0,1,1,0,1,2,0,0,1,1,1,2,2,1,1,1]
labels = {}
for index, value in enumerate(true_labels):
    if value not in labels.keys():
        labels[value] = [index]
    else:
        labels.get(value).append(index)
for value in labels.values():
    print(value)
    random.shuffle(value)
print(list(labels.keys()))

# def label_drawing(self, X, ):
#     y = self.predict(X)
#     iris = load_iris()
#     X = iris.data
#     targets = iris.target_names
#     features = iris.feature_names
#     plt.figure(figsize=(10, 4))
#     plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs', label=targets[0])
#     plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'kx', label=targets[1])
#     plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], 'ro', label=targets[2])
#     plt.xlabel(features[2])
#     plt.ylabel(features[3])
#     plt.legend()
#     plt.show()

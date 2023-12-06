from sklearn.datasets import load_iris
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import random
# def load_data(data_raw,true_labels,num=1,random_seed=2023):
#     random.seed(random_seed)
#     labels = {}
#     for index, value in enumerate(true_labels):
#         if value not in labels.keys():
#             labels[value] = [index]
#         else:
#             labels.get(value).append(index)
#     for value in labels.values():
#         random.shuffle(value)
#     label = []
#     for i in range(num):
#         for value in labels.values():
#             label.append(value[i])
#     # print(label)
#     # print(random_unlabeled_points)
#     mixed_labels = deepcopy(true_labels)
#     # print(mixed_labels)
#     for i in range(len(mixed_labels)):
#         # print(i)
#         if i not in label:
#             mixed_labels[i] = -1
#     # for i in range(len(y_mixed)):
#     #     if y_mixed[i] == -1:
#     #         continue
#     #     if rng.random() < noise_rate:  # 将有标签的样本中的noise_rate替换为随机标签
#     #         candidate_ids = np.random.permutation(n_class).tolist()
#     #         candidate_ids.remove(y_mixed[i])
#     #         y_mixed[i] = candidate_ids[0]
#     ss = StandardScaler()
#     data_raw = ss.fit_transform(data_raw)
#     return data_raw, true_labels, mixed_labels
import numpy as np
import pandas as pd
def load_titanic(path):
    data = pd.read_csv(path, header=0)
    data = data.drop(columns=['PassengerId', 'Name', 'Sex' ,'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    # data.drop(axis=0)
    survived = data['Survived']
    survived = survived.to_numpy()
    data = data.drop('Survived', axis=1)
    data = data.to_numpy()
    return data, survived


if __name__ == '__main__' :
    # data_raw, true_labels = load_iris(return_X_y=True)
    # x,y,y_mixed = load_data(data_raw,true_labels,2)
    # print(y_mixed)

    # a = np.loadtxt(open(path, 'rb'), delimiter=",", skiprows=1)
    path = 'train.csv'
    data = pd.read_csv(path, header = 0)
    data = data.drop(columns=['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
    # data.drop(axis=0)
    survived = data['Survived']
    survived = survived.to_numpy()
    data= data.drop('Survived',axis=1)
    data = data.to_numpy()


    iris_data,iris_label = load_iris(return_X_y=True)
    print(survived)
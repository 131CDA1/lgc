import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from copy import deepcopy
'''
加载数据集，进行归一化，随机删除样本标签
'''
def load_excel(path):
    full_data = pd.read_excel(path)
    true_labels = full_data.iloc[:,-1]
    data = full_data.iloc[:,0:-1]
    data = data.to_numpy()
    true_labels = true_labels.to_numpy()
    data_zscore = (data - data.mean(axis=0))/data.std(axis=0)
    data_maxmin = (data - data.min())/(data.max() - data.min())
    return data, true_labels

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
    mixed_labels = deepcopy(true_labels)
    for i in range(len(mixed_labels)):
        # print(i)
        if i not in label:
            mixed_labels[i] = -1
    ss = StandardScaler()
    data_raw = ss.fit_transform(data_raw)
    return data_raw, true_labels, mixed_labels, labels, label, k

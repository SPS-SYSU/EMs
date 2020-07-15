import pandas as pd
import numpy as np
import math as mt
import torch
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
import torch.utils.data as Data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
fr = open("F:/data/20191116data.csv")  ###这个位置你们要成电脑中相应文件的位置
#"E:/20191022_220.csv"
df = pd.DataFrame(pd.read_csv(fr,header=None))

ans_1 = df.iloc[:, 0]  # 第一列参考答案
data_1 = df.iloc[:, 2:]  # 对应第一列答案的数据集
merge_ans1_data1 = pd.concat([ans_1, data_1], axis=1).dropna(axis=0,
                                                             how='any')  ##合并分子式，第一列参考答案，以及相应的数据集#并删除掉含有Nan的行
Y_1 = np.mat(merge_ans1_data1.iloc[:, 0])
X_1 = np.mat(merge_ans1_data1.iloc[:, 1:])
X_1_min = np.mat(X_1.min(0))
X_1_max = np.mat(X_1.max(0))
fenmu_1 = np.tile(X_1_max - X_1_min, (np.shape(X_1)[0], 1))
Y_1_revised = []
for i in range(np.shape(Y_1)[1]):
    Y_1_revised.append(float(Y_1[0, i]))
Y_1_revised = [0 if i > 2.6 else 1 for i in Y_1_revised]

X_1_revised = np.multiply(X_1 - X_1_min, 1 / fenmu_1)

Pca = PCA(n_components=44).fit(X_1_revised)#109，220
X_1_revised = Pca.transform(X_1_revised)
x_zhou = np.arange(0, 44)#109 220
X_zero = [i for (i, v) in enumerate(Y_1_revised) if v == 0]
X_one = [i for (i, v) in enumerate(Y_1_revised) if v == 1]
X_data_one = [X_1_revised[i] for i in X_one]
X_data_zero = [X_1_revised[i] for i in X_zero]
X_one_sum = np.mean(X_data_one, axis=0).T
X_zero_sum = np.mean(X_data_zero, axis=0).T
X_one_sum = pd.DataFrame(X_one_sum)
X_zero_sum = pd.DataFrame(X_zero_sum)


X_temp0 = [i for i in X_zero_sum[0]]
X_temp1 = [i for i in X_one_sum[0]]
print('pca {}'.format(44))#109 220
total = 0
for i,j in zip(X_temp0, X_temp1):
    total += np.power((i-j),2)#每一项的平方
print("len {}".format(len(X_temp1)))
print("var {}".format(total/len(X_temp1)))
print("std {}".format(np.power(total/len(X_temp1),0.5)))
plt.ylabel("Average of dimensions values")
plt.xlabel("Dimensions")
plt.plot(x_zhou, X_temp1, 'r', label='High affinity')
plt.plot(x_zhou, X_temp0, 'b', label='Low affinity')
print()
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
plt.legend(prop = font1)


plt.show()
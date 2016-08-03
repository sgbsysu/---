# -*- coding:utf-8 -*-
# 数据预处理
# 对原始数据的缺失值进行分析
# 统计每个变量缺失值的数目并可视化
# 将每个样本的缺失值数量作为一维新的特征，并将其离散化

#第二种，统计每个特征缺失值的数量，并可视化
#将缺失值抄过一定阈值的特征删除
#对剩余的特征进行均值填充

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

train_x_csv = '../data/train_x.csv'
train_y_csv = '../data/train_y.csv'
test_x_csv = '../data/test_x.csv'
train_unlabeled_csv = '../data/train_unlabeled.csv'
features_type_csv = '../data/features_type.csv'

#计算每个特征的缺失值数量
def countNullByFeature():
    train_x = pd.read_csv(train_x_csv)
    train_x_feature_null = (train_x < 0).sum()
    train_x_feature_null = train_x_feature_null.sort_values()
    train_x_feature_null.to_csv('../data/train_x_feature_null.csv')

    test_x = pd.read_csv(test_x_csv)
    test_x_feature_null = (test_x < 0).sum()
    test_x_feature_null = test_x_feature_null.sort_values()
    test_x_feature_null.to_csv('../data/test_x_feature_null.csv')

    train_unlabeled = pd.read_csv(train_unlabeled_csv)
    train_unlabeled_feature_null = (train_unlabeled < 0).sum()
    train_unlabeled_feature_null = train_unlabeled_feature_null.sort_values()
    train_unlabeled_feature_null.to_csv('../data/train_unlabeled_feature_null.csv')

def visualFeatureNull():
    train_x = pd.read_csv('../data/train_x_feature_null.csv')
    y = train_x['0'].values
    x = range(len(y))
    plt.figure(1)
    plt.scatter(x,y,c='k')
    plt.title('train set')
    # plt.show()

    test_x = pd.read_csv('../data/test_x_feature_null.csv')
    y2 = test_x['0'].values
    x2 = range(len(y2))
    plt.figure(2)
    plt.scatter(x2,y2,c='k')
    plt.title('test set')
    # plt.show()

    train_unlabeled = pd.read_csv('../data/train_unlabeled_feature_null.csv')
    y3 = train_unlabeled['0'].values
    x3 = range(len(y3))
    plt.figure(3)
    plt.scatter(x3,y3,c='k')
    plt.title('train_unlabeled set')
    plt.show()

#根据属性缺失值的绘制图可得
#train:y>=6000
#test:y>=2000
#train_unlabel:y>=24000
def deleteFeature():
    train_x_feature = pd.read_csv('../train_x_feature_null.csv')
    train_x_feature_to_delete = list(train_x_feature[train_x_feature['0']<6000].uid)

    features_type = pd.read_csv(features_type_csv)
    features_numeric = list(features_type[features_type.type == 'numeric'].feature)
    features_category = list(features_type[features_type.type == 'category'].feature)
    features_n = [f for f in features_numeric if f in train_x_feature_to_delete]
    features_c = [f for f in features_category if f in train_x_feature_to_delete]
    types_n = ['numeric']*len(features_n)
    types_c = ['category']*len(features_c)
    features_n.extend(features_c)
    types_n.extend(types_c)
    features_type_del = pd.DataFrame(columns=['feature','type'])
    features_type_del.feature = features_n
    features_type_del.type = types_n
    features_type_del.to_csv('../data/features_type_del.csv',index=None)


    train_x = pd.read_csv(train_x_csv)
    train_x_delete_feature = pd.DataFrame(train_x.uid,columns=[['uid']+train_x_feature_to_delete])
    for feature in train_x_feature_to_delete:
        train_x_delete_feature[feature] = train_x[feature]
    train_x_delete_feature.to_csv('../data/train_x_delete_feature.csv',index=None)

    test_x_feature = pd.read_csv('../test_x_feature_null.csv')
    test_x_feature_to_delete = list(test_x_feature[test_x_feature['0']<2000].uid)
    test_x = pd.read_csv(test_x_csv)
    test_x_delete_feature = pd.DataFrame(test_x.uid,columns=[['uid']+test_x_feature_to_delete])
    for feature in test_x_feature_to_delete:
        test_x_delete_feature[feature] = test_x[feature]
    test_x_delete_feature.to_csv('../data/test_x_delete_feature.csv',index=None)

    train_unlabeled_feature = pd.read_csv('../train_unlabeled_feature_null.csv')
    train_unlabeled_feature_to_delete = list(train_unlabeled_feature[train_unlabeled_feature['0']<24000].uid)
    train_unlabeled = pd.read_csv(train_unlabeled_csv)
    train_unlabeled_delete_feature = pd.DataFrame(train_unlabeled.uid,columns=[['uid']+train_unlabeled_feature_to_delete])
    for feature in train_unlabeled_feature_to_delete:
        train_unlabeled_delete_feature[feature] = train_unlabeled[feature]
    train_unlabeled_delete_feature.to_csv('../data/train_unlabeled_delete_feature.csv',index=None)

#统计每个样本，缺失值的数目
def countNull():
    #将每行样本的缺失值存储在train_x_null['n_null']中
    train_x = pd.read_csv(train_x_csv)
    train_x['n_null'] = (train_x < 0).sum(axis=1)
    train_x.to_csv('../data/train_x_null.csv',index=None,encoding='utf-8')
    #对test_x,train_unlabeled进行同样的统计
    test_x = pd.read_csv(test_x_csv)
    test_x['n_null'] = (test_x < 0).sum(axis=1)
    test_x.to_csv('../data/test_x_null.csv',index=None,encoding='utf-8')
    train_unlabeled = pd.read_csv(train_unlabeled_csv)
    train_unlabeled['n_null'] = (train_unlabeled < 0).sum(axis=1)
    train_unlabeled.to_csv('../data/train_unlabeled_null.csv',index=None,encoding='utf-8')

#对样本缺失值进行可视化处理
def visualNull():
    train_x = pd.read_csv('../data/train_x_null.csv')[['uid','n_null']]
    test_x = pd.read_csv('../data/test_x_null.csv')[['uid','n_null']]
    train_unlabeled = pd.read_csv('../data/train_unlabeled_null.csv')[['uid','n_null']]

    train_x = train_x.sort(columns='n_null')
    test_x = test_x.sort(columns='n_null')
    train_unlabeled = train_unlabeled.sort(columns='n_null')
    y_train_x = train_x.n_null.values
    x_train_x = range(len(y_train_x))
    # plt.subplot(311)
    plt.scatter(x_train_x,y_train_x,c='k',color='green')
    plt.title('train set')
    plt.show()
    # plt.subplot(312)
    y_test_x = test_x.n_null.values
    x_test_x = range(len(y_test_x))
    plt.scatter(x_test_x,y_test_x,c='k',color='blue')
    plt.title('test set')
    plt.show()
    # plt.subplot(313)
    y_train_unlabeled = train_unlabeled.n_null.values
    x_train_unlabeled = range(len(y_train_unlabeled))
    plt.scatter(x_train_unlabeled,y_train_unlabeled,c='k',color='red',label='train_unlabeled')
    plt.title('train unlabeled')
    plt.show()

#根据缺失值所得到的散点图，对其进行分段处理
#查看散点图，可估计：
#0~35；35~69；69~145；145~190；190~
def discreteNull():
    train_x = pd.read_csv('../data/train_x_null.csv')
    train_x['discrete_null'] = train_x.n_null
    train_x.discrete_null[train_x.discrete_null<=35] = 1
    train_x.discrete_null[(train_x.discrete_null>35)&(train_x.discrete_null<=69)] = 2
    train_x.discrete_null[(train_x.discrete_null>69)&(train_x.discrete_null<=145)] = 3
    train_x.discrete_null[(train_x.discrete_null>145)&(train_x.discrete_null<=190)] = 4
    train_x.discrete_null[train_x.discrete_null>190] = 5
    train_x.to_csv('../data/train_x_discrete_null.csv',index=None,encoding='utf=8')

    test_x = pd.read_csv('../data/test_x_null.csv')
    test_x['discrete_null'] = test_x.n_null
    test_x.discrete_null[test_x.discrete_null<=35] = 1
    test_x.discrete_null[(test_x.discrete_null>35)&(test_x.discrete_null<=69)] = 2
    test_x.discrete_null[(test_x.discrete_null>69)&(test_x.discrete_null<=145)] = 3
    test_x.discrete_null[(test_x.discrete_null>145)&(test_x.discrete_null<=190)] = 4
    test_x.discrete_null[test_x.discrete_null>190] = 5
    test_x.to_csv('../data/test_x_discrete_null.csv',index=None,encoding='utf-8')

    train_unlabeled = pd.read_csv('../data/train_unlabeled_null.csv')
    train_unlabeled['discrete_null'] = train_unlabeled.n_null
    train_unlabeled.discrete_null[train_unlabeled.discrete_null<=35] = 1
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>35)&(train_unlabeled.discrete_null<=69)] = 2
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>69)&(train_unlabeled.discrete_null<=145)] = 3
    train_unlabeled.discrete_null[(train_unlabeled.discrete_null>145)&(train_unlabeled.discrete_null<=190)] = 4
    train_unlabeled.discrete_null[train_unlabeled.discrete_null>190] = 5
    train_unlabeled.to_csv('../data/train_unlabeled_discrete_null.csv',index=None,encoding='utf-8')

# def run():
#     # countNull()
#     # visualNull()
#     # discreteNull()
#     # countNullByFeature()
#     # visualFeatureNull()
#     deleteFeature()
# deleteFeature()
visualNull()
# visualFeatureNull()


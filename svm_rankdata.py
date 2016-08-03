# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
import os
import random

if (os.path.exists('../result_svm_M1_thera') == False):
    os.mkdir('../result_svm_M1_thera')

#加载文件路径
train_x_rank = '../data/train_x_numeric_rank.csv'
train_y_csv = '../data/train_y.csv'
test_x_rank = '../data/test_x_numeric_rank.csv'

def loadData():
    train_x = pd.read_csv(train_x_rank)
    train_y = pd.read_csv(train_y_csv)
    train_x = train_x.drop(['uid'],axis=1)
    train_y = train_y.y
    #对数据进行归一化
    train_x_norm = preprocessing.normalize(train_x)
    # train_y = train_xy.y

    test = pd.read_csv(test_x_rank)
    test_uid = test.uid
    test_x = test.drop('uid',axis=1)
    #对数据进行归一化
    test_x_norm = preprocessing.normalize(test_x)
    y = list(train_y.values)

    return train_x_norm,y,test_x_norm,test_uid

def predict_SVM():
    X,y,test_x,test_uid = loadData()
    model = SVC(C=2.5,gamma=0.03,kernel='rbf',probability=True,random_state=23333)
    model.fit(X,y)
    test_y = model.predict_proba(test_x)
    result = pd.DataFrame(columns=["uid","score"])
    result.uid=test_uid
    result.score = test_y[:,1]
    result.to_csv('../result/result_SVM_rank.csv',index=None,encoding='utf-8')
predict_SVM()
# def average_result():
#     files = os.listdir('../result_svm_M1_thera')
#     pred = pd.read_csv('../result_svm_M1_thera/' + files[0])

#     uid = pred.uid
#     score = pred.score

#     for f in files[1:]:
#         pred = pd.read_csv('../result_svm_M1_thera/'+f)
#         score += pred.score
#     avg_score = score/len(files)

#     result_svm_M1 = pd.DataFrame(uid,columns=["uid"])
#     result_svm_M1["score"] = avg_score

#     result_svm_M1.to_csv('../result_svm_M1_thera/result_svm_M1_thera.csv',index=None,encoding='utf-8')

# def pipeLine(iteration,C,gamma,random_seed):
#     print iteration
#     X,y,test_x,test_uid = loadData()
#     model = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,random_state=random_seed)
#     model.fit(X,y)
#     pred = model.predict_proba(test_x)
#     test_result = pd.DataFrame(columns=["uid","score"])
#     test_result.uid=test_uid
#     test_result.score = pred[:,1]
#     test_result.to_csv('../result_svm_M1_thera/svm_pred{0}.csv'.format(iteration),index=None,encoding='utf-8')

# def testSVM():
#     random_seed = range(1024,2048)
#     C = [i/10 for i in range(40,60)]
#     gamma = [i/1000.0 for i in range(31,51)]
#     random.shuffle(random_seed)
#     random.shuffle(C)
#     random.shuffle(gamma)
#     for i in range(20):
#         pipeLine(i,C[i],gamma[i],random_seed[i])
#     average_result()


# testSVM()






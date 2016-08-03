import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
import csv
import xgboost as xgb

def dd(data):
    return data.strip().split('"')[1]

def loadDataSet(fileName):
    # featNum = len(open(fileName).readline().split(','))-1
    dataSet = []
    userID = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        lineArr = []
        if curLine[0] == '"uid"':
            continue
        index = 0
        for item in curLine:
            if index == 0:
                userID.append(item)
                index = index+1
            if item == '':
                lineArr.append(NaN)
            if '"' in item:
                lineArr.append(float(dd(item)))
            else:
                lineArr.append(float(item))
        dataSet.append(lineArr)
    return dataSet,userID

def dealDataSet():
    imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
    train_x,l1 = loadDataSet('../data/train_x.csv')
    imp.fit(np.array(train_x))
    _train_x = imp.transform(np.array(train_x))

    train_y = np.genfromtxt('../data/train_y.csv',delimiter=',')[1:,1]

    test_x,result_id = loadDataSet('../data/test_x.csv')
    imp.fit(np.array(test_x))
    _test_x = imp.transform(np.array(test_x))

    train_x_normalized = preprocessing.normalize(_train_x,norm='l2')
    test_x_normalized = preprocessing.normalize(_test_x,norm='l2')

    return train_x_normalized,train_y,test_x_normalized,result_id
# def test():
#     train_x,train_y,test_x,user_id = dealDataSet()
#     clf = svm.SVC()
#     clf.fit(train_x,train_y)
#     res = clf.predict(test_x)
#     fw = open('../data/result3.csv','w')
#     index = 0
#     fw.write('"uid"'+','+'"score"'+'\n')
#     for item in res:
#         fw.write(str(user_id[index])+','+str(int(item))+'\n')
#         index = index+1
def testXGB():
    train_x,train_y,test_x,user_id = dealDataSet()
    params = {
        'booster':'gbtree',
        'objective':'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight':2000.0/12348.0,
        'eval_metric':'auc',
        'gamma':0.3,
        'max_depth':5,
        'lambda':550,
        'seed':23333,
        'nthread':5
    }
    bst = xgb.train()
    fw = open('../data/result3.csv','w')
    index = 0
    fw.write('"uid"'+','+'"score"'+'\n')
    for item in res:
        fw.write(str(user_id[index])+','+str(int(item))+'\n')
        index = index+1
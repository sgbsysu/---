import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
def test():
    train_x = np.genfromtxt('../data/train_x.csv',delimiter=',')[1:,1:]
    train_y = np.genfromtxt('../data/train_y.csv',delimiter=',')[1:,1]
    imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit(np.array(train_x))
    train_x = imp.transform(np.array(train_x))
    clf_l1_LR = SVC(kernel='rbf',C=2.5,gamma=0.03,probability=True)
    clf_l1_LR.fit(train_x,train_y)

    test_x = np.genfromtxt('../data/test_x.csv',delimiter=',')
    test_x = test_x[1:,1:]
    imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
    imp.fit(np.array(test_x))
    test_x = imp.transform(np.array(test_x))
    indexArr = test_x[1:,0]
    res = clf_l1_LR.predict_proba(test_x)
    # return indexArr
    writer = csv.writer(open('../data/result1.csv','wb'))
    writer.writerow(["uid","score"])
    index = 0
    for item in res:
        resultrow = [indexArr[index],item]
        writer.writerow(resultrow)
        index = index + 1
test()

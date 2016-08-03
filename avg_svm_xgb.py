# -*- coding:utf-8 -*-

#计算SVM模型预测的结果平均值
import pandas as pd 
import os

if (os.path.exists('../result_svm_xgb') == False):
    os.mkdir('../result_svm_xgb')

result_svm_M1 = pd.read_csv('../result/svm_0.618.csv')
result_xgb = pd.read_csv('../result/0.7204.csv')
uid = result_svm_M1.uid
result_svm_xgb = pd.DataFrame(uid,columns=["uid"])
result_svm_xgb["score"] = 0.400*result_svm_M1.score+0.60*result_xgb.score+0.0001*(abs(result_svm_M1.score - result_xgb.score))

result_svm_xgb.to_csv('../result_svm_xgb/result_svm_xgb_0.60_0.40.csv',index=None,encoding='utf-8')
# -*- coding:utf-8 -*-
# 特征选择
# 利用决策树的思想，分别对原始特征、排序特征、离散特征的重要性进行排序

#使用交叉验证
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import os

if (os.path.exists('../model') == False):
    os.mkdir('../model')
if (os.path.exists('../result') == False):
    os.mkdir('../result')

#set data path
train_x_csv = '../data/train_x_discrete.csv'
train_y_csv = '../data/train_y.csv'
test_x_csv = '../data/test_x_discrete.csv'
features_type = '../data/features_type.csv'

#load data
train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x,train_y,on='uid')

test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'],axis=1)

#split train set, generate train,val,test set
train_xy = train_xy.drop(['uid'],axis=1)
train,val = train_test_split(train_xy,test_size=0.3,random_state=1)
y = train.y
x = train.drop(['y'],axis=1)
val_y = val.y
val_x = val.drop(['y'],axis=1)

#xgboost start 

dtest = xgb.DMatrix(test_x)
dtrain = xgb.DMatrix(x,label=y)
dval = xgb.DMatrix(val_x,label=val_y)

params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':100,
    'scale_pos_weight':1500.0/13458.0,
    'max_depth':10,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':2.5,
    'eval_metric':'auc',
    'gamma':0.2,
    'lambda':500,
    'eta':0.005,
    'seed':23333,
    'nthread':8
}

watchlist = [(dtrain,'train'),(dval,'val')]
model = xgb.train(params,dtrain,num_boost_round=30000,evals=watchlist)
model.save_model('../model/xgb.model')

test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv('../result/test_result_xgb_discrete.csv',index=None,encoding='utf-8')

#save feature score and feature infomation
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(),key=lambda x:x[1],reverse=True)
fs_key = []
fs_value = []
for (key,value) in feature_score:
    # 
    fs_key.append(key)
    fs_value.append(value)
feature_score_csv = pd.DataFrame(columns=['feature','score'])
feature_score_csv.feature = fs_key
feature_score_csv.score = fs_value
feature_score_csv.to_csv('../data/discrete_feature_score.csv',index=None,encoding='utf-8')



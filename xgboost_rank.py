# -*- coding:utf-8 -*-
#使用xgboost进行预测
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import os

train_x_numeric_rank_csv = '../data/train_x_numeric_rank.csv'
train_x = pd.read_csv(train_x_numeric_rank_csv)

test_x_numeric_rank_csv = '../data/test_x_numeric_rank.csv'
test_x = pd.read_csv(test_x_numeric_rank_csv)

test_uid = test_x.uid
test_x = test_x.drop(['uid'],axis=1)

train_y_csv = '../data/train_y.csv'
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x,train_y,on='uid')


train_xy = train_xy.drop(['uid'],axis=1)

train,val = train_test_split(train_xy,test_size=0.3,random_state=1)
y = train.y
x = train.drop(['y'],axis=1)
val_y = val.y
val_x = val.drop(['y'],axis=1)

dtest = xgb.DMatrix(test_x)
dtrain = xgb.DMatrix(x,label=y)
dval = xgb.DMatrix(val_x,label=val_y)

params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':100,
    'scale_pos_weight':0.025,
    'max_depth':10,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':2.5,
    'eval_metric':'auc',
    'gamma':0.1,
    'lambda':500,
    'eta':0.003,
    'seed':233333,
    'nthread':8
}
watchlist = [(dtrain,'train'),(dval,'val')]
model = xgb.train(params,dtrain,num_boost_round=10000,evals=watchlist)
# model.save_model('../model_2/xgb_M1.model')

test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv('../result/result_xgb_rank.csv',index=None,encoding='utf-8')
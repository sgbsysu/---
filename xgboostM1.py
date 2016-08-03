# -*- coding:utf-8 -*-
#使用xgboost进行预测
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import os

if (os.path.exists('../data_model') == False):
    os.mkdir('../data_model')

#首先加载计数特征数据与缺失值特征数据
train_x_ndiscrete = '../data/train_x_ndiscrete.csv'
train_x_discrete_null = '../data/train_x_discrete_null.csv'

train_x_ndiscrete = pd.read_csv(train_x_ndiscrete)
train_x_discrete_null = pd.read_csv(train_x_discrete_null)[['uid','n_null','discrete_null']]

#合并数据
train_x_n_null = pd.merge(train_x_ndiscrete,train_x_discrete_null,on='uid')

test_x_ndiscrete = '../data/test_x_ndiscrete.csv'
test_x_discrete_null = '../data/test_x_discrete_null.csv'

test_x_ndiscrete = pd.read_csv(test_x_ndiscrete)
test_x_discrete_null = pd.read_csv(test_x_discrete_null)[['uid','n_null','discrete_null']]

#合并数据
test_x_n_null = pd.merge(test_x_ndiscrete,test_x_discrete_null,on='uid')

train_unlabeled_ndiscrete = '../data/train_unlabeled_ndiscrete.csv'
train_unlabeled_discrete_null = '../data/train_unlabeled_discrete_null.csv'

train_unlabeled_ndiscrete = pd.read_csv(train_unlabeled_ndiscrete)
train_unlabeled_discrete_null = pd.read_csv(train_unlabeled_discrete_null)[['uid','n_null','discrete_null']]

#合并数据
train_unlabeled_n_null = pd.merge(train_unlabeled_ndiscrete,train_unlabeled_discrete_null,on='uid')

#处理原始特征、排序特征、离散特征，提取前600个数据
#读取特征重要性排序后的特征文件
raw_feature_score_csv = '../data/raw_feature_score.csv'
rank_feature_score_csv = '../data/rank_feature_score.csv'
discrete_feature_score_csv = '../data/discrete_feature_score.csv'

raw_feature_score = pd.read_csv(raw_feature_score_csv)
rank_feature_score = pd.read_csv(rank_feature_score_csv)
discrete_feature_score = pd.read_csv(discrete_feature_score_csv)

len_raw = int(len(raw_feature_score)/2)
len_rank = int(len(rank_feature_score)/2)
len_discrete = int(len(discrete_feature_score)/2)

selected_raw_feature = list(raw_feature_score.feature[0:len_raw])
selected_rank_feature = list(rank_feature_score.feature[0:len_rank])
selected_discrete_feature = list(discrete_feature_score.feature[0:len_discrete])

train_x_raw_csv = '../data/train_x.csv'
train_x_numeric_rank_csv = '../data/train_x_numeric_rank.csv'
train_x_numeric_discrete_csv = '../data/train_x_discrete.csv'

train_x_raw = pd.read_csv(train_x_raw_csv)[['uid']+selected_raw_feature]
train_x_numeric_rank = pd.read_csv(train_x_numeric_rank_csv)[['uid']+selected_rank_feature]
train_x_numeric_discrete = pd.read_csv(train_x_numeric_discrete_csv)[['uid']+selected_discrete_feature]

#数据集合并
train_x_raw_rank = pd.merge(train_x_raw,train_x_numeric_rank,on='uid')
train_x_raw_rank_discrete = pd.merge(train_x_raw_rank,train_x_numeric_discrete,on='uid')

train_x = pd.merge(train_x_raw_rank_discrete,train_x_n_null,on='uid')

test_x_raw_csv = '../data/test_x.csv'
test_x_numeric_rank_csv = '../data/test_x_numeric_rank.csv'
test_x_numeric_discrete_csv = '../data/test_x_discrete.csv'

test_x_raw = pd.read_csv(test_x_raw_csv)[['uid']+selected_raw_feature]
test_x_numeric_rank = pd.read_csv(test_x_numeric_rank_csv)[['uid']+selected_rank_feature]
test_x_numeric_discrete = pd.read_csv(test_x_numeric_discrete_csv)[['uid']+selected_discrete_feature]

#数据集合并
test_x_raw_rank = pd.merge(test_x_raw,test_x_numeric_rank,on='uid')
test_x_raw_rank_discrete = pd.merge(test_x_raw_rank,test_x_numeric_discrete,on='uid')

test_x = pd.merge(test_x_raw_rank_discrete,test_x_n_null,on='uid')

#写入test文件
# test_x.to_csv('../data_model/test_x.csv',index=None)

test_uid = test_x.uid
test_x = test_x.drop(['uid'],axis=1)

train_unlabeled_raw_csv = '../data/train_unlabeled.csv'
train_unlabeled_numeric_rank_csv = '../data/train_unlabeled_numeric_rank.csv'
train_unlabeled_numeric_discrete_csv = '../data/train_unlabeled_discrete.csv'

train_unlabeled_raw = pd.read_csv(train_unlabeled_raw_csv)[['uid']+selected_raw_feature]
train_unlabeled_numeric_rank = pd.read_csv(train_unlabeled_numeric_rank_csv)[['uid']+selected_rank_feature]
train_unlabeled_numeric_discrete = pd.read_csv(train_unlabeled_numeric_discrete_csv)[['uid']+selected_discrete_feature]

#数据集合并
train_unlabeled_raw_rank = pd.merge(train_unlabeled_raw,train_unlabeled_numeric_rank,on='uid')
train_unlabeled_raw_rank_discrete = pd.merge(train_unlabeled_raw_rank,train_unlabeled_numeric_discrete,on='uid')

train_unlabeled = pd.merge(train_unlabeled_raw_rank_discrete,train_unlabeled_n_null,on='uid')

train_y_csv = '../data/train_y.csv'
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x,train_y,on='uid')
#将特征选择后的数据进行存储，以便进行其他单模型训练
# train_xy.to_csv('../data_model/train_xy.csv',index=None)


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
    'scale_pos_weight':0.05,
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
model = xgb.train(params,dtrain,num_boost_round=25000,evals=watchlist)
model.save_model('../model/xgb_M1.model')

test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv('../result/result_xgb_M1.csv',index=None,encoding='utf-8')









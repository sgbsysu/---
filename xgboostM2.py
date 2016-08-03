# -*- coding:utf-8 -*-
#使用xgboost进行预测
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import os
import random

if (os.path.exists('../data_model_M2_1') == False):
    os.mkdir('../data_model_M2_1')
if (os.path.exists('../result_M2_1') == False):
    os.mkdir('../result_M2_1')

#首先加载计数特征数据与缺失值特征数据
train_x_ndiscrete = '../data/train_x_ndiscrete.csv'
train_x_discrete_null = '../data/train_x_discrete_null.csv'

train_x_ndiscrete = pd.read_csv(train_x_ndiscrete)
train_x_discrete_null = pd.read_csv(train_x_discrete_null)[['uid','n_null','discrete_null']]

#合并数据
train_x_n_null = pd.merge(train_x_ndiscrete,train_x_discrete_null,on='uid')
n_null_features = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discrete_null']

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

# len_raw = int(len(raw_feature_score)/2)
# len_rank = int(len(rank_feature_score)/2)
# len_discrete = int(len(discrete_feature_score)/2)

# selected_raw_feature = list(raw_feature_score.feature[0:len_raw])
# selected_rank_feature = list(rank_feature_score.feature[0:len_rank])
# selected_discrete_feature = list(discrete_feature_score.feature[0:len_discrete])

train_x_raw_csv = '../data/train_x.csv'
train_x_numeric_rank_csv = '../data/train_x_numeric_rank.csv'
train_x_numeric_discrete_csv = '../data/train_x_discrete.csv'

train_x_raw = pd.read_csv(train_x_raw_csv)
train_x_numeric_rank = pd.read_csv(train_x_numeric_rank_csv)
train_x_numeric_discrete = pd.read_csv(train_x_numeric_discrete_csv)

#数据集合并
train_x_raw_rank = pd.merge(train_x_raw,train_x_numeric_rank,on='uid')
train_x_raw_rank_discrete = pd.merge(train_x_raw_rank,train_x_numeric_discrete,on='uid')

train_x = pd.merge(train_x_raw_rank_discrete,train_x_n_null,on='uid')

test_x_raw_csv = '../data/test_x.csv'
test_x_numeric_rank_csv = '../data/test_x_numeric_rank.csv'
test_x_numeric_discrete_csv = '../data/test_x_discrete.csv'

test_x_raw = pd.read_csv(test_x_raw_csv)
test_x_numeric_rank = pd.read_csv(test_x_numeric_rank_csv)
test_x_numeric_discrete = pd.read_csv(test_x_numeric_discrete_csv)

#数据集合并
test_x_raw_rank = pd.merge(test_x_raw,test_x_numeric_rank,on='uid')
test_x_raw_rank_discrete = pd.merge(test_x_raw_rank,test_x_numeric_discrete,on='uid')

test_x = pd.merge(test_x_raw_rank_discrete,test_x_n_null,on='uid')

#写入test文件
# test_x.to_csv('../data_model/test_x.csv',index=None)

test_uid = test_x.uid
test_x_ = test_x.drop(['uid'],axis=1)

# train_unlabeled_raw_csv = '../data/train_unlabeled.csv'
# train_unlabeled_numeric_rank_csv = '../data/train_unlabeled_numeric_rank.csv'
# train_unlabeled_numeric_discrete_csv = '../data/train_unlabeled_discrete.csv'

# train_unlabeled_raw = pd.read_csv(train_unlabeled_raw_csv)[['uid']+selected_raw_feature]
# train_unlabeled_numeric_rank = pd.read_csv(train_unlabeled_numeric_rank_csv)[['uid']+selected_rank_feature]
# train_unlabeled_numeric_discrete = pd.read_csv(train_unlabeled_numeric_discrete_csv)[['uid']+selected_discrete_feature]

# #数据集合并
# train_unlabeled_raw_rank = pd.merge(train_unlabeled_raw,train_unlabeled_numeric_rank,on='uid')
# train_unlabeled_raw_rank_discrete = pd.merge(train_unlabeled_raw_rank,train_unlabeled_numeric_discrete,on='uid')

# train_unlabeled = pd.merge(train_unlabeled_raw_rank_discrete,train_unlabeled_n_null,on='uid')

train_y_csv = '../data/train_y.csv'
train_y = pd.read_csv(train_y_csv)
train_xy_ = pd.merge(train_x,train_y,on='uid')
#将特征选择后的数据进行存储，以便进行其他单模型训练
# train_xy.to_csv('../data_model/train_xy.csv',index=None)
# def testt():
#     selected_raw_feature = list(raw_feature_score.feature[0:500])
#     selected_rank_feature = list(rank_feature_score.feature[0:500])
#     selected_discrete_feature = list(discrete_feature_score.feature[0:100])
#     return selected_raw_feature,selected_rank_feature,selected_discrete_feature,n_null_features

def pipeline(iteration,random_seed,feature_num,rank_feature_num,discrete_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    selected_raw_feature = list(raw_feature_score.feature[0:feature_num])
    selected_rank_feature = list(rank_feature_score.feature[0:rank_feature_num])
    selected_discrete_feature = list(discrete_feature_score.feature[0:discrete_feature_num])
    train_xy = train_xy_[selected_raw_feature+selected_rank_feature+selected_discrete_feature+n_null_features+['y']]
    test_x = test_x_[selected_raw_feature+selected_rank_feature+selected_discrete_feature+n_null_features]
    # x = train_xy.drop(['y'],axis=1)
    # y = train_xy.y

    # train,val = train_test_split(train_xy,test_size=0.3,random_state=1)
    # y = train.y
    # x = train.drop(['y'],axis=1)
    # val_y = val.y
    # val_x = val.drop(['y'],axis=1)
    x = train_xy.drop(['y'],axis=1)
    y = train_xy.y

    dtest = xgb.DMatrix(test_x)
    dtrain = xgb.DMatrix(x,label=y)
    # dval = xgb.DMatrix(val_x,label=val_y)

    params = {
        'booster':'gbtree',
        'objective':'binary:logistic',
        'early_stopping_rounds':100,
        'scale_pos_weight':0.05,
        'max_depth':max_depth,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight,
        'eval_metric':'auc',
        'gamma':gamma,
        'lambda':lambd,
        'eta':0.003,
        'seed':random_seed,
        'nthread':8
    }
    watchlist = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=20000,evals=watchlist)
    model.save_model('../data_model_M2_1/xgb_M2_{0}.model'.format(iteration))

    test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv('../result_M2_1/result_xgb_M2_{0}.csv'.format(iteration),index=None,encoding='utf-8')

def test():
    random_seed = range(1000,2000,10)
    feature_num = range(500,700,2)
    rank_feature_num = range(500,700,2)
    discrete_feature_num = range(500,700,2)
    gamma = [i/10000.0 for i in range(500,700,2)]
    lambd = range(500,700,2)
    max_depth = [8,10,12]
    subsample = [i/1000.0 for i in range(500,700,2)]
    colsample_bytree = [i/1000.0 for i in range(250,350,1)]
    min_child_weight = [i/1000.0 for i in range(250,550,3)]

    random.shuffle(random_seed)
    random.shuffle(feature_num)
    random.shuffle(rank_feature_num)
    random.shuffle(discrete_feature_num)
    random.shuffle(gamma)
    random.shuffle(lambd)
    random.shuffle(max_depth)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)

    for i in range(0,5):
        pipeline(i,random_seed[i],feature_num[i],rank_feature_num[i],discrete_feature_num[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])
    average_result()

def average_result():
    files = os.listdir('../result_M2_1')
    pred = pd.read_csv('../result_M2_1/' + files[0])

    uid = pred.uid
    score = pred.score

    for f in files[1:]:
        pred = pd.read_csv('../result_M2_1/'+f)
        score += pred.score
    avg_score = score/len(files)

    result_svm_M1 = pd.DataFrame(uid,columns=["uid"])
    result_svm_M1["score"] = avg_score

    result_svm_M1.to_csv('../result/result_xgb_M2_1.csv',index=None,encoding='utf-8')

test()




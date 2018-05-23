# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
import time
import warnings

warnings.filterwarnings("ignore")

train_file = './data/training2.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')

training = data_set[feature_list].values


kf = KFold(n_splits = 5,random_state=2017,shuffle=True)
rmse_list = []
for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index],training[val_index],label[val_index]

    params = {
            'task': 'train','boosting_type': 'gbdt','objective': 'regression',
            'metric': {'l2', 'rmse'},'max_depth':5,'num_leaves':21,'n_estimators':1500,
            'min_data_in_leaf':300,'learning_rate': 0.02,
            'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
            'num_boost_round':1500,
            'verbose': -2}

    lgb_train = lgb.Dataset(X_train,label=y_train,feature_name=feature_list)
    lgb_eval = lgb.Dataset(X_val,label=y_val,feature_name=feature_list, reference=lgb_train)
    gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval,early_stopping_rounds=100)
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print("rmse:",rmse)
    rmse_list.append(rmse)

print("kflod rmse: {}\n mean rmse : {}".format(rmse_list, np.mean(np.array(rmse_list))))
















import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,HuberRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:] 
                mse = mean_squared_error(y_holdout,y_pred)  
                print("mse",mse)             

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='neg_mean_squared_error')
        print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

# rf params
rf_params = {}
rf_params['n_estimators'] = 120
rf_params['max_depth'] = 8
rf_params['min_samples_split'] = 5
rf_params['min_samples_leaf'] = 2


# lgb params
lgb_params = {}
lgb_params['n_estimators'] = 120
# lgb_params['max_bin'] = 2
lgb_params['learning_rate'] = 0.01 # shrinkage_rate
lgb_params['metric'] = 'mae'          # or 'mae'
lgb_params['sub_feature'] = 0.6    
lgb_params['subsample'] = 0.8 # sub_row
lgb_params['num_leaves'] = 11        # num_leaf
lgb_params['min_data'] = 5         # min_data_in_leaf
lgb_params['verbose'] = -1
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3


# XGB model
xgb_model = XGBRegressor(n_estimators=120)

gbdt_model = GradientBoostingRegressor()

# lgb model
lgb_model = LGBMRegressor(**lgb_params)

# RF model
rf_model = RandomForestRegressor(**rf_params)

# ET model
et_model = ExtraTreesRegressor()

# SVR model
# SVM is too slow in more then 10000 set
# svr_model = SVR(kernel='rbf', C=5.0, epsilon=0.005)

# hr_model = HuberRegressor()

# lr_model = LinearRegression()

# DecsionTree model
dt_model = DecisionTreeRegressor()

# AdaBoost model
ada_model = AdaBoostRegressor()


if os.path.exists('../data/tree_train_data.pkl'):
        train_y = pickle.load(open('../data/train_y.pkl','rb'))
        test_ID = pickle.load(open('../data/test_ID.pkl','rb'))
        tree_train_data = pickle.load(open('../data/tree_train_data.pkl','rb'))
        tree_test_data = pickle.load(open('../data/tree_test_data.pkl','rb'))
else:
    tree_train_data,train_y, tree_test_data, test_ID = make_training_data('../训练.xlsx','../测试A.xlsx')

stack_tree = Ensemble(n_splits=5,
        stacker=LinearRegression(),
        base_models=(gbdt_model,lgb_model,rf_model, xgb_model, et_model, ada_model))

y_test = stack_tree.fit_predict(tree_train_data, train_y, tree_test_data)

# sub_df = pd.read_csv('../github.csv',header=None)
# sub_df.columns=['ID','lr']
# lr_pred = np.array(sub_df['lr'])
# tree_pred = 0.6*y_test + 0.4*lr_pred

data = {'ID':test_ID,'Y':y_test}
pred_df = pd.DataFrame(data)
 
sub_df = pd.read_csv('../测试A-答案模板.csv',header=None)
sub_df.columns=['ID']
sub_df = pd.merge(sub_df,pred_df,how='left',on='ID')
sub_df.to_csv('merge8.csv',header=None,index=False)
sub_df['Y'].plot(kind='kde')
plt.show()


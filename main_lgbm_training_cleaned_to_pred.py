"""
generating predictions based on cleaned csv input files

data cleaning and preprocessing part can be used to non-regression and non-
neural network models

use main_lgbm_training_raw_to_pred.py or main_feature_generation_save_local.py
to create features
"""

import pandas as pd
import gc

from sklearn.model_selection import train_test_split

import lightgbm as lgb

print(">>> Defining model")
params = {}
params['objective'] = 'binary'
params['boosting'] = 'gbdt'
params['verbose'] = 0
params['num_leaves'] = 256
params['learning_rate'] = 0.2
params['bagging_fraction'] = 0.95
params['bagging_freq'] = 1
params['bagging_seed'] = 1122
params['metric'] = 'auc'
params['max_depth'] = 8
boost_rounds = 100


print(">>> Preparing data for lgbm training...")
train = pd.read_csv("../raw_data/full_train.csv")
test = pd.read_csv("../raw_data/full_test.csv")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


y = train['target'].values
X = train.drop(['target'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
d_train = lgb.Dataset(X_train, y_train)
d_val = lgb.Dataset(X_valid, y_valid)
watchlist = [d_val]

ids = test['id'].values
X_test = test.drop(['id'], axis=1)

del train, test
gc.collect()

print('>>> Training LGBM model...')
model = lgb.train(params, train_set=d_train, valid_sets=watchlist,
                  verbose_eval=10, num_boost_round=boost_rounds)
model_name = 'lgbm_model_lr_' + str(params['learning_rate']) + '_leaves_' +\
    str(params['num_leaves']) + '.txt'
model.save_model(model_name)

print('>>> Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm_name = 'lgbm_submit_lr_' + str(params['learning_rate']) + '_leaves_' +\
    str(params['num_leaves']) + '.csv.gz'
subm.to_csv(subm_name, compression='gzip', index=False, float_format='%.5f')
print('Done!')

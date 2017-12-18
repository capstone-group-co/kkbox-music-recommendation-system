"""
generating predictions based on clean csv input and sklearn GridSearchCV with
hyperparameters for Light GBM.
Optimal parameters as of now:
num_leaves = 256
max_depths = 9
learning_rate = 0.3
subsamples = 1

to test:
reg_alphas = [0, 0.05, 0.1]
reg_lambdas = [0, 0.05, 0.1]
"""

import pandas as pd
import gc

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

print(">>> Defining hyper parameters")
learning_rate = 0.3
num_leaves = 256
max_depth = 9
subsamples = [1, 0.9, 0.8, 0.7]

print(">>> Preparing data for lgbm training...")
train = pd.read_csv("../raw_data/full_train.csv")
test = pd.read_csv("../raw_data/full_test.csv")

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


y = train['target'].values
X = train.drop(['target'], axis=1)

ids = test['id'].values
X_test = test.drop(['id'], axis=1)

del train, test
gc.collect()

print('>>> Training LGBM model...')
estimator = lgb.LGBMClassifier(objective='binary',
                               num_leaves=num_leaves,
                               learning_rate=learning_rate,
                               random_state=1122,
                               max_depth=max_depth,
                               n_jobs=5,
                               n_estimators=100)

param_grid = {
    'subsample': subsamples
}

gbm = GridSearchCV(estimator, param_grid, verbose=1)
gbm.fit(X, y)
best_params = gbm.best_params_
print(">>> Best parameters: %s" % best_params)

print('>>> Making predictions and saving them...')
p_test = gbm.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm_name = 'lgbm_submit_leaves' + str(num_leaves) + '.csv.gz'
subm.to_csv(subm_name, compression='gzip', index=False, float_format='%.5f')
print('Done!')

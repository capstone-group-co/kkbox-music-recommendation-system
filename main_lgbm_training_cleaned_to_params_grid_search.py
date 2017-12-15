"""
generating predictions based on raw csv input files from end to end
"""

import pandas as pd
import gc

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

print(">>> Defining hyper parameters")
max_depths = [8, 9]
num_leaves = 256
learning_rates = [0.2, 0.8]

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
estimator = lgb.LGBMClassifier(object='binary',
                               num_leaves=num_leaves,
                               random_state=1122,
                               n_jobs=2,
                               n_estimators=100)

param_grid = {
    'learning_rate': learning_rates,
    'max_depth': max_depths
}

gbm = GridSearchCV(estimator, param_grid, verbose=1)
gbm.fit(X, y)
best_params = gbm.best_params_
print(">>> Best parameters: %s" % best_params)

print(">>> Feature Importances:")
print(X.columns)
print(gbm.feature_importances_)

print('>>> Making predictions and saving them...')
p_test = gbm.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm_name = 'lgbm_submit_leaves' + str(num_leaves) + '.csv.gz'
subm.to_csv(subm_name, compression='gzip', index=False, float_format='%.5f')
print('Done!')

"""
Strategy of this script:
Load preprocessed data from sqlite db in large chunks ->
Train lgbm model with the batch ->
Save the lgbm model in local directory ->
Load another large batch ->
Load the saved lgbm model and continue training

This script is used only for sqlite batch training. The script streamline is:
main_feature_generation_save_local.py ->
main_csv_to_sqlite.py ->
elvin_lgbm_sqlite_training_batch_loop.py ->
elvin_lgbm_sqlite_testing_batch_loop.py

This model does not have optimal result, probably due to following reasons:
1. Input data has many low performing SVD results (from songs table)
2. Training a tree structure with GBM in batches will not find the optimal
splits for the entire dataset

The second reason is more likely. Therefore, I will focus on reinventing the
whole data processing -> data merging -> training process
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# define data loading parameters
print('>>> Defining data loading params...')
sqlite_url = 'sqlite:///../kkbox.db'
train_table = 'full_train'
test_table = 'full_test'
kkbox_engine = create_engine(sqlite_url)
kkbox_conn = kkbox_engine.connect()
chunksize = 100000
train_len = kkbox_conn.execute(
    "SELECT max(rowid) FROM full_train").fetchone()[0]

# define lgbm training parameters
print(">>> Defining lgbm training params...")
boost_rounds = 100
params = {}
params['objective'] = 'binary'
params['boosting'] = 'gbdt'
params['verbose'] = 0
params['num_leaves'] = 100
params['learning_rate'] = 0.2
params['bagging_fraction'] = 0.95
params['bagging_freq'] = 1
params['bagging_seed'] = 1122
params['metric'] = 'auc'


def create_db(train_batch):
    """create lgbm dataset for training, train_batch should be np.ndarray"""
    x_train, x_test, y_train, y_test = train_test_split(
        train_batch[:, 1:], train_batch[:, 0], test_size=0.1,
        random_state=1122)
    d_train = lgb.Dataset(x_train, y_train)
    d_test = lgb.Dataset(x_test, y_test)
    watchlist = [d_test]
    print(">>> gbm training set created")
    return x_test, y_test, d_train, watchlist


# training loop for 1 epoch
print(">>> Start lgbm training iteration")
i = 0
iteration = 1
while i < train_len:
    print(">>> Iteration %i..." % iteration)
    stmt = ('SELECT * FROM %s WHERE rowid >= %i' % (train_table, i + 1))
    trainloader = pd.read_sql(stmt, kkbox_conn, chunksize=chunksize)
    train_chunk = next(iter(trainloader))
    train_chunk = train_chunk.as_matrix()
    x_test, y_test, d_train, watchlist = create_db(train_chunk)
    if iteration == 1:  # first batch does not have existing model
        model = lgb.train(params, verbose_eval=boost_rounds, train_set=d_train,
                          valid_sets=watchlist, num_boost_round=boost_rounds)
    else:
        model = lgb.train(params, train_set=d_train, valid_sets=watchlist,
                          verbose_eval=boost_rounds, init_model='model.txt',
                          num_boost_round=boost_rounds)
    model.save_model('model.txt')
    i += chunksize  # load next batch of data from db
    iteration += 1  # increment iteration count

epoch = 0
model_name = 'lgbm_model_lr_' + str(params['learning_rate']) + '_leaves_' +\
    str(params['num_leaves']) + '_n_' + str(chunksize) + '_epoch_' +\
    str(epoch + 1) + '.txt'
model.save_model(model_name)

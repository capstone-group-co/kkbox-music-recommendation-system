"""
Strategy of this script:
Load data and model -> Generate predictions in batches ->
Concatenate predictions and save to local directory

This script is used only for sqlite batch testing. The script streamline is:
main_feature_generation_save_local.py ->
main_csv_to_sqlite.py ->
elvin_lgbm_sqlite_training_batch_loop.py ->
elvin_lgbm_sqlite_testing_batch_loop.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine

# define data loading parameters
print('>>> Defining data loading params...')
sqlite_url = 'sqlite:///../kkbox.db'
test_table = 'full_test'
kkbox_engine = create_engine(sqlite_url)
kkbox_conn = kkbox_engine.connect()
chunksize = 10000
model_file = "lgbm_model_lr_0.2_leaves_100_n_100000_epoch_1.txt"
test_len = kkbox_conn.execute(
    "SELECT max(rowid) FROM full_test").fetchone()[0]


# training loop for 1 epoch
print(">>> Starting lgbm testing iteration...")
i = 0
iteration = 1
pred = np.array([])
ids = np.array([])
print(">>> Loading trained lgbm model...")
bst = lgb.Booster(model_file=model_file)
while i < test_len:
    print(">>> Iteration %i..." % iteration)
    stmt = ('SELECT * FROM %s WHERE rowid >= %i' % (test_table, i + 1))
    testloader = pd.read_sql(stmt, kkbox_conn, chunksize=chunksize)
    test_chunk = next(iter(testloader)).as_matrix()
    x_test = test_chunk[:, 1:]
    ids = np.append(ids, test_chunk[:, 0])
    pred = np.append(pred, bst.predict(x_test))
    i += chunksize  # load next batch of data from db
    iteration += 1  # increment iteration count
    print(">>> %i predictions made: %.2f%% completed" % (
        len(pred), len(pred)/test_len))

print(">>> Prediction completed. Total predictions: %i" % len(pred))
epoch = 0
submission = pd.DataFrame()
submission['id'] = ids.astype(int)
submission['target'] = pred
output_name = 'lgbm_submission_n_' + str(chunksize) + '_epoch_' +\
    str(epoch + 1) + '.csv.gz'
print(">>> Saving predictions to local file")
submission.to_csv(output_name, compression='gzip',
                  index=False, float_format='%.5f')
print('Done!')

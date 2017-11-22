"""
Script to transfer final csv feature set into sqlite db for data streaming
Be sure to customize root_path to match actual csv file directory
Streamline:
main_feature_generation_save_local.py -->
main_csv_to_sqlite.py (this file) -->
model training scripts that streams data from sqlite
"""
import numpy as np
import pandas as pd
import time
from sqlalchemy import create_engine

print(">>> define file paths")
root_path = '../raw_data'
train_file = '/'.join([root_path, 'full_train.csv'])
test_file = '/'.join([root_path, 'full_test.csv'])

print(">>> glimpse files")
print(pd.read_csv(train_file, nrows=2))
print(pd.read_csv(test_file, nrows=2))

print(">>> link to sqlite database kkbox.db")
kkbox_db = create_engine('sqlite:///kkbox.db')

print(">>> load full_train.csv into sqlite")
chunksize = 100000
i = 0
j = 1
train_start = time.time()
print(">>> itering through full_train.csv")
for df in pd.read_csv(train_file, chunksize=chunksize, iterator=True):
    df = df.rename(columns={c: c.replace(" ", '_') for c in df.columns})
    df.index += j  # iterator starts by 1, then jumps to where left last round
    df = df.drop(['Unnamed:_0'], axis=1)
    i += 1
    df.to_sql('full_train', kkbox_db, if_exists='append', index=False)
    j = df.index[-1] + 1  # get the last index of current load
    print(">>> iter %i: successful" % i)
train_end = time.time()
print(">>> loading full_train.csv took %i seconds" % (train_end - train_start))

print(">>> load full_test.csv into sqlite")
chunksize = 100000
i = 0
j = 1
test_start = time.time()
print(">>> itering through full_test.csv")
for df in pd.read_csv(test_file, chunksize=chunksize, iterator=True):
    df = df.rename(columns={c: c.replace(" ", '_') for c in df.columns})
    df.index += j
    df = df.drop(['Unnamed:_0'], axis=1)
    i += 1
    df.to_sql('full_test', kkbox_db, if_exists='append', index=False)
    j = df.index[-1] + 1
    print(">>> iter %i: successful" % i)
test_end = time.time()
print(">>> loading full_test.csv took %i seconds" % (test_end - test_start))

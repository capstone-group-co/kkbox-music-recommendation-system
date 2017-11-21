"""
testing the csv_to_sqlite code shows that the elvin_csv_to_sqlite.py file has
no problems the mismatched row happened in the data transformation stage
"""

import numpy as np
import pandas as pd
import time
from sqlalchemy import create_engine

print(">>> define file paths")
root_path = '../../test_data'
train_file = '/'.join([root_path, 'train.csv'])
test_file = '/'.join([root_path, 'test.csv'])


def get_dataset_length(file_path, had_header=True):
    """Retrieve file length of a large csv file"""
    with open(file_path, 'r') as f:
        length = 0
        for _ in f:
            length += 1
        length = length - had_header
    return length


train_length = get_dataset_length(train_file)
print("test train csv has %i rows" % train_length)
test_length = get_dataset_length(test_file)
print("test test csv has %i rows" % test_length)


print(">>> link to sqlite database test.db")
test_db = create_engine('sqlite:///test.db')

print(">>> load test train.csv into sqlite")
chunksize = 1000
i = 0
j = 1
print(">>> itering through test train.csv")
for df in pd.read_csv(train_file, chunksize=chunksize, iterator=True):
    df = df.rename(columns={c: c.replace(" ", '_') for c in df.columns})
    df.index += j  # iterator starts by 1, then jumps to where left last round
    df['index'] = df.index
    i += 1
    df.to_sql('train', test_db, if_exists='append', index=False)
    j = df.index[-1] + 1  # get the last index of current load
    print(">>> iter %i: successful" % i)

print(">>> load test test.csv into sqlite")
chunksize = 1000
i = 0
j = 1
print(">>> itering through test test.csv")
for df in pd.read_csv(test_file, chunksize=chunksize, iterator=True):
    df = df.rename(columns={c: c.replace(" ", '_') for c in df.columns})
    df.index += j
    df['index'] = df.index
    i += 1
    df.to_sql('test', test_db, if_exists='append', index=False)
    j = df.index[-1] + 1
    print(">>> iter %i: successful" % i)


def get_table_length(conn, table_name):
    length = conn.execute(
        'SELECT max(rowid) FROM ' + table_name).scalar()
    return length


sqlite_train_length = get_table_length(test_db, 'train')
print("test sqlite train has %i rows" % sqlite_train_length)
sqlite_test_length = get_table_length(test_db, 'test')
print("test sqlite train has %i rows" % sqlite_test_length)

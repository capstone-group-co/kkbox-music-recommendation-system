import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load files and report basic information about dataframes

file_names = ['raw_data/members.csv',
              'raw_data/song_extra_info.csv',
              'raw_data/songs.csv',
              'raw_data/test.csv',
              'raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

files = [members, song_extra_info, songs, test, train]


def describe_df(df, df_name):
    var_names = " ".join(df.columns.values.astype('U'))
    print("""
The dataframe {} has {} columns/variables and {} observations.
The dataframe comes with the following variables:
{}
    """.format(
        df_name, df.shape[1], df.shape[0], var_names), end='')

for x, y in zip(files, file_names):
    describe_df(x, y)


# In depth examinination of data types, data values, and data integrity
# Check the data types of members
print(members.dtypes)
"""
msno                      object
city                       int64
bd                         int64
gender                    object
registered_via             int64
registration_init_time     int64
expiration_date            int64
dtype: object
"""
members.city = members.city.astype('category')
members.loc[np.logical_or(members.bd <= 0, members.bd > 100), 'bd'] = np.nan
members.bd.astype('int64')

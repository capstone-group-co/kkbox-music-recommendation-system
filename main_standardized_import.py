import numpy as np
import pandas as pd
import datetime


def describe_df(df, df_name):
    dtypes = df.dtypes.to_frame(name='dtypes').T
    n_null = df.isnull().sum()[df.isnull().sum() != 0].\
        to_frame(name='nulls').T
    print("""
$$$ The dataframe {} has {} columns/variables and {} observations.
    The dataframe comes with the following variables:
{}
    The following variables have null values:
{}
    """.format(
        df_name, df.shape[1], df.shape[0], dtypes, n_null), end='')


def kkbox_cleaning(members, song_extra_info, songs, test, train):
    print(">>> DataFrame cleaning initiated:")
    members.city = members.city.astype('category')
    members.loc[np.logical_or(members.bd <= 0, members.bd > 100), 'bd'] =\
        np.nan
    members.bd = members.bd.astype(float)
    members.gender = members.gender.astype('category')
    members.registered_via = members.registered_via.astype('category')
    members.registration_init_time = pd.to_datetime(
        members.registration_init_time.astype(str))
    members.expiration_date = pd.to_datetime(
        members.expiration_date.astype(str))
    members.expiration_date = members.\
        expiration_date.replace(datetime.datetime(1970, 1, 1), np.nan)
    print(">>> members dataframe is cleaned")
    print(">>> song_extra_info is cleaned")
    songs.song_length = songs.song_length.astype(float)
    songs.language = songs.language.astype('category')
    print(">>> songs is cleaned")
    test.id = test.id.astype(str)
    test.source_system_tab = test.source_system_tab.astype('category')
    test.source_screen_name = test.source_screen_name.astype('category')
    test.source_type = test.source_type.astype('category')
    print(">>> test is cleaned")
    train.source_system_tab = train.source_system_tab.astype('category')
    train.source_screen_name = train.source_screen_name.astype('category')
    train.source_type = train.source_type.astype('category')
    print(">>> train is cleaned")
    return members, song_extra_info, songs, test, train


# Load files, clean up and report basic information about dataframes
if __name__ == "__main__":
    file_names = ['raw_data/members.csv',
                  'raw_data/song_extra_info.csv',
                  'raw_data/songs.csv',
                  'raw_data/test.csv',
                  'raw_data/train.csv']

    members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                    for x in file_names]

    members, song_extra_info, songs, test, train =\
        kkbox_cleaning(members, song_extra_info, songs, test, train)

    files = [members, song_extra_info, songs, test, train]

    for x, y in zip(files, file_names):
        describe_df(x, y)

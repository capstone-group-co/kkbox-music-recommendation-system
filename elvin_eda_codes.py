import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

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

# DATA CLEANING STEPS

# members dataframe
# members.msno is the unique member id
print("Is members.msno unique id for members set? {}".format(
    len(members) == len(members.msno.unique())
))

# members.city is a categorical variable
members.city = members.city.astype('category')
print("The members.city variable includes {} unique levels.".format(
    len(members.city.unique())))

# members.db is a numerical variable with positive integers
# that falls within sensible human age range
print("Originally, members.bd variable has the following unique values: {}".
      format(" ".join(np.sort(members.bd.unique()).astype(str))))
members.loc[np.logical_or(members.bd <= 0, members.bd > 100), 'bd'] = np.nan
members.bd = members.bd.astype(float)
print("After cleaning, the members.bd variable now has values: {}".format(
    " ".join(np.sort(members.bd.unique()).astype(str))))

# members.gender is a categorical variable
members.gender = members.gender.astype('category')
print("The members.gender variable includes {} levels.".format(
    members.gender.unique()))

# members.registered_via is a categorical variable
members.registered_via = members.registered_via.astype('category')
print("The members.registered_via variable includes {} levels.".format(
    len(members.registered_via.unique())))

# members.registration_init_time is a datetime variable in %Y%M%d
members.registration_init_time = pd.to_datetime(
    members.registration_init_time.astype(str))
print("The members.registration_init_time variable is a date array with:\n\
{}".format(
    members.registration_init_time.describe()))

# members.expiration_date also is a datetime variable in %Y%M%d
members.expiration_date = pd.to_datetime(
    members.expiration_date.astype(str))
print("The members.expiration_date variable is a date array with:\n\
{}".format(
    members.expiration_date.describe()))
# It appears that a user's membership date is 1970-01-01, which is
# equivalent to N/A in this context. Let's check the record:
print("The only member without a valid expiration_date is:\n\
{}".format(
    members.loc[(members.expiration_date < datetime.datetime(2000, 1, 1))]))
# expiration_date is ealier than registration_init_time, it is invalid
# replace with np.nan
members.expiration_date = members.\
    expiration_date.replace(datetime.datetime(1970, 1, 1), np.nan)
# Recheck the summary of the column
print("The members.expiration_date variable is a date array with:\n\
{}".format(
    members.expiration_date.describe()))

# Reprint the dtype of members table
print("After data cleaning, the members table variables are as below:\n\
{}".format(members.dtypes))

# song_extra_info dataframe
# All three variables seem to be okay with original object dtype
print("The song_extra_info table variables are as below:\n\
{}".format(song_extra_info.dtypes))

# songs dataframe
# song_id is the unique key of this table
print("Is songs.song_id unique id for songs set? {}".format(
    len(songs) == len(songs.song_id.unique())))

# song_length is the length of each song in ms
songs.song_length = songs.song_length.astype(float)
print("The song_length has the follow characteristics: \n\
{}".format(songs.song_length.describe()))

# TODO: decide on how to unpack genre_ids

# TODO: decide on how to unpack artist_name

# TODO: decide on how to unpack composer

# TODO: decide on how to unpack lyricist

# language is a factor that indicates song language
songs.language = songs.language.astype('category')
print("The language has the {} levels.".format(len(songs.language.unique())))

# test dataframe (same for train I suppose)
# test.id is the unique identifier for the test table
test.id = test.id.astype(str)
print("Is test.id unique id for test set? {}".format(
    len(test) == len(test.id.unique())))

# test.msno is the foreign key to link to members table
print("The test set covers {} members, {} of which \
are not from the members set".format(
    len(test.msno.unique()),
    sum(np.logical_not(pd.Series(test.msno.unique()).isin(
        members.msno.unique())))))

# test.song_id is the foreign key to link to songs table
print("The test set covers {} songs, {} of which \
are not from the songs set".format(
    len(test.song_id.unique()),
    sum(np.logical_not(pd.Series(test.song_id.unique()).isin(
        songs.song_id.unique())))))

# test.source_system_tab is a categorical variable
test.source_system_tab = test.source_system_tab.astype('category')
print("The source_system_tab has {} levels.".format(
    len(test.source_system_tab.unique())))

# test.source_screen_name is a categorical variable
test.source_screen_name = test.source_screen_name.astype('category')
print("The source_screen_name has {} levels.".format(
    len(test.source_screen_name.unique())))

# test.source_type is a categorical variable
test.source_type = test.source_type.astype('category')
print("The source_type has {} levels.".format(
    len(test.source_type.unique())))

# reprint the dtypes of test set
print("After data cleaning, the test table variables are as below:\n\
{}".format(test.dtypes))

# Repeat the same cleaning process for train set
# train.msno is the foreign key to link to members table
print("The train set covers {} members, {} of which \
are not from the members set".format(
    len(train.msno.unique()),
    sum(np.logical_not(pd.Series(train.msno.unique()).isin(
        members.msno.unique())))))

# train.song_id is the foreign key to link to songs table
print("The train set covers {} songs, {} of which \
are not from the songs set".format(
    len(train.song_id.unique()),
    sum(np.logical_not(pd.Series(train.song_id.unique()).isin(
        songs.song_id.unique())))))

# train.source_system_tab is a categorical variable
train.source_system_tab = train.source_system_tab.astype('category')
print("The source_system_tab has {} levels.".format(
    len(train.source_system_tab.unique())))

# train.source_screen_name is a categorical variable
train.source_screen_name = train.source_screen_name.astype('category')
print("The source_screen_name has {} levels.".format(
    len(train.source_screen_name.unique())))

# train.source_type is a categorical variable
train.source_type = train.source_type.astype('category')
print("The source_type has {} levels.".format(
    len(train.source_type.unique())))

# train.target is a dummy variable
print("The target variable is summarized as below:\n\
{}".format(train.target.describe()))

# reprint the dtypes of test set
print("After data cleaning, the train table variables are as below:\n\
{}".format(train.dtypes))

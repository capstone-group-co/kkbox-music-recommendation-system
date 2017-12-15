"""
generating predictions based on raw csv input files from end to end

data cleaning and preprocessing part can be used to non-regression and non-
neural network models, since data has not been scaled or binarized or
normalized

script also contains greyed out scripts that saves processed dataframe to a csv
"""

import numpy as np
import pandas as pd
import datetime
import main_standardized_import as stanimp
import gc

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
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

print(">>> Loading data")
file_names = ['../raw_data/members.csv',
              '../raw_data/song_extra_info.csv',
              '../raw_data/songs.csv',
              '../raw_data/test.csv',
              '../raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

members, song_extra_info, songs, test, train =\
    stanimp.kkbox_cleaning(members, song_extra_info, songs, test, train)

# SONG EXTRA INFO FEATURE ENGINEERING
print(">>> SONG EXTRA INFO FEATURE ENGINEERING <<<")


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

song_extra_info['song_year'] = song_extra_info['isrc'].apply(isrc_to_year)
song_extra_info.drop(['isrc', 'name'], axis=1, inplace=True)


# SONGS TABLE FEATURE ENGINEERING
print(">>> SONGS TABLE FEATURE ENGINEERING <<<")
# matrix factorization on song meta data
print(">>> Starting matrix factorization (SVD) for songs meta data...")
song_id = songs.song_id.tolist()
artist_names = songs['artist_name'].str.split('&').tolist()
genre_ids = songs['genre_ids'].str.split('|').tolist()
composers = songs['composer'].str.split('|').tolist()
lyricists = songs['lyricist'].str.split('|').tolist()
meta_list = []
for i in range(len(song_id)):
    meta_dict = {}
    if not isinstance(artist_names[i], list):
            pass
    else:
        for artist_name in artist_names[i]:
            meta_dict[artist_name.strip()] = 1
    if not isinstance(genre_ids[i], list):
        pass
    else:
        for genre_id in genre_ids[i]:
            meta_dict[genre_id.strip()] = 1
    if not isinstance(composers[i], list):
        pass
    else:
        for composer in composers[i]:
            meta_dict[composer.strip()] = 1
    if not isinstance(lyricists[i], list):
        pass
    else:
        for lyricist in lyricists[i]:
            meta_dict[lyricist.strip()] = 1
    meta_list.append(meta_dict)
vec_meta = DictVectorizer().fit(meta_list)
m_meta = vec_meta.transform(meta_list)
svd_components = 30
rand_seed = 1122
svd_meta = TruncatedSVD(n_components=svd_components,
                        algorithm='randomized',
                        n_iter=5,
                        random_state=rand_seed).fit(m_meta)
print(">>> Meta data SVDs explained %.2f%% of total variance."
      % (svd_meta.explained_variance_.sum() * 100))
v_meta = svd_meta.transform(m_meta)
meta_svds = ['meta_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(meta_svds):
    songs[name] = v_meta[:, i]

songs = songs.drop(
    ['genre_ids', 'artist_name', 'composer', 'lyricist'],
    axis=1)

print(">>> Songs table feature engineering completed!")
del song_id, artist_names, genre_ids, composers, lyricists, meta_list,\
    meta_dict, vec_meta, m_meta, svd_meta, v_meta, meta_svds
gc.collect()

# MEMBERS TABLE FEATURE ENGINEERING
print(">>> MEMBERS TABLE FEATURE ENGINEERING <<<")
# create day_since_founding
print(">>> Generating day_since_founding for membership init time")
launch_time = datetime.datetime.strptime('2004-10-01', '%Y-%m-%d')
members['day_since_founding'] = ((
    members.registration_init_time -
    launch_time)/np.timedelta64(1, 'D')).astype(float)

print(">>> Generating day_membership_duration before membership expires")
# create day_membership_duration
members['day_membership_duration'] = ((
    members.expiration_date - members.registration_init_time) / np.timedelta64(
    1, 'D')).astype(float)
dmd_imputer = Imputer().fit(
    members.day_membership_duration.values.reshape(-1, 1))
members['day_membership_duration'] = dmd_imputer.transform(
    members.day_membership_duration.values.reshape(-1, 1))

print(">>> Imputing bd")
# imputing bd
bd_imputer = Imputer(strategy='most_frequent').fit(
    members.bd.values.reshape(-1, 1))
members['bd'] = bd_imputer.transform(members.bd.values.reshape(-1, 1))

members = members.drop(['registration_init_time', 'expiration_date'], axis=1)
print(">>> Members table feature engineering completed!")
del launch_time, dmd_imputer, bd_imputer
gc.collect()

# MERGING TABLES
print(">>> Generating final tables...")
print(">>> Merging train with songs and members")
train = train.merge(songs, how='left', on='song_id').merge(
    members, how='left', on='msno').merge(
    song_extra_info, on='song_id', how='left')
print(">>> Train now has %i variables and %i observations" %
      (train.shape[1], train.shape[0]))
print(">>> Merging test with songs and members")
test = test.merge(songs, how='left', on='song_id').merge(
    members, how='left', on='msno').merge(
    song_extra_info, on='song_id', how='left')
print(">>> Test now has %i variables and %i observations" %
      (test.shape[1], test.shape[0]))

train = train.drop(['msno', 'song_id'], axis=1)
# train.to_csv('full_train.csv', float_format='%.8f', index=False)
test = test.drop(['msno', 'song_id'], axis=1)
# test.to_csv('full_test.csv', float_format='%.8f', index=False)

del songs, members
gc.collect()

print(">>> Preparing data for lgbm training")
y = train['target'].values
X = train.drop(['target'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
d_train = lgb.Dataset(X_train, y_train)
d_val = lgb.Dataset(X_valid, y_valid)

ids = test['id'].values
X_test = test.drop(['id'], axis=1)

del train, test
gc.collect()

watchlist = [d_val]

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

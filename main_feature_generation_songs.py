import numpy as np
import pandas as pd
import main_standardized_import as stanimp
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD

file_names = ['../raw_data/members.csv',
              '../raw_data/song_extra_info.csv',
              '../raw_data/songs.csv',
              '../raw_data/test.csv',
              '../raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

members, song_extra_info, songs, test, train =\
    stanimp.kkbox_cleaning(members, song_extra_info, songs, test, train)

del members, song_extra_info, test, train

print(">>> start binarize songs language")
# binarize language categories
language_lb = LabelBinarizer()
language_lb.fit(songs.language.values.astype(int).reshape(-1, 1))
lang_bi_v = language_lb.transform(
    songs.language.values.astype(int).reshape(-1, 1))
lang_binaries = ['language_bi_'+str(i+1) for i in range(lang_bi_v.shape[1])]
for i, name in enumerate(lang_binaries):
    songs[name] = lang_bi_v[:, i]

# matrix factorization on song artist_name
print(">>> start matrix factorization for songs artist_name")
# create list of list for the multilabel variable
artist_names = songs['artist_name'].str.split('&').tolist()
# create list of unique identifier for later assembling
song_id = songs.song_id.tolist()
# generate dictionary for the sklearn DictVectorizer
dict_artist = [{name.strip(): 1 for name in artist_names[i]} for
               i in range(len(song_id))]
# Train a DictVectorizer based on d_artist_name
vec_artist = DictVectorizer().fit(dict_artist)
# Generate a CSR sparse matrix use the trained DictVectorizer
m_artist = vec_artist.transform(dict_artist)
# Train a TruncatedSVD model with m_artist_name
svd_components = 10
rand_seed = 1122
svd_artist_name = TruncatedSVD(n_components=svd_components,
                               algorithm='randomized',
                               n_iter=5,
                               random_state=rand_seed).fit(m_artist)
# Generate a svd matrix for song artist_names
v_artist = svd_artist_name.transform(m_artist)
# Assemble a pd.DataFrame with the artist_svd
# artist_df can be merged into songs df and replace the artist_names column
artist_svds = ['artist_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(artist_svds):
    songs[name] = v_artist[:, i]

# matrix factorization on genre_ids
print(">>> start matrix factorization for songs genre_ids")
genre_ids = songs['genre_ids'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_genre = [{} if not isinstance(genre_ids[i], list)
              else {genre_id.strip(): 1 for genre_id in genre_ids[i]}
              for i in range(len(genre_ids))]
vec_genre = DictVectorizer().fit(dict_genre)
m_genre = vec_genre.transform(dict_genre)
svd_components = 10
rand_seed = 1122
svd_genre_id = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_genre)
v_genre = svd_genre_id.transform(m_genre)
genre_svds = ['genre_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(genre_svds):
    songs[name] = v_genre[:, i]

# matrix factorization on song composer
print(">>> start matrix factorization for songs composer")
composers = songs['composer'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_composer = [{} if not isinstance(composers[i], list)
                 else {composer.strip(): 1 for composer in composers[i]}
                 for i in range(len(composers))]
vec_composer = DictVectorizer().fit(dict_composer)
m_composer = vec_composer.transform(dict_composer)
svd_components = 10
rand_seed = 1122
svd_composer = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_composer)
v_composer = svd_composer.transform(m_composer)
composer_svds = ['composer_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(composer_svds):
    songs[name] = v_composer[:, i]

# matrix factorization on lyricist
print(">>> start matrix factorization for songs lyricist")
lyricists = songs['lyricist'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_lyricist = [{} if not isinstance(lyricists[i], list)
                 else {lyricist.strip(): 1 for lyricist in lyricists[i]}
                 for i in range(len(lyricists))]
vec_lyricist = DictVectorizer().fit(dict_lyricist)
m_lyricist = vec_lyricist.transform(dict_lyricist)
svd_components = 10
rand_seed = 1122
svd_lyricist = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_lyricist)
v_lyricist = svd_lyricist.transform(m_lyricist)
lyricist_svds = ['lyricist_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(lyricist_svds):
    songs[name] = v_lyricist[:, i]

songs = songs.drop(
    ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language'],
    axis=1)

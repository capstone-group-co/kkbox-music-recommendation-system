import numpy as np
import pandas as pd
import gensim
import standardized_import as stanimp

file_names = ['raw_data/members.csv',
              'raw_data/song_extra_info.csv',
              'raw_data/songs.csv',
              'raw_data/test.csv',
              'raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

members, song_extra_info, songs, test, train =\
    stanimp.kkbox_cleaning(members, song_extra_info, songs, test, train)

# generate song_id-genre_id matrix
# code will likely take much portion of RAM
songs_genre_long = songs[['song_id', 'genre_ids']].astype(str)
songs_genre_long = pd.DataFrame(songs_genre_long.genre_ids.str.split('|').
                                tolist(),
                                index=songs_genre_long.song_id).stack()
songs_genre_long = songs_genre_long.reset_index()[[0, 'song_id']]
songs_genre_long.columns = ['genre_ids', 'song_id']
songs_genre_long['count'] = 1
songs_genre_wide = songs_genre_long.pivot(index='song_id',
                                          columns='genre_ids',
                                          values='count')
del songs_genre_long

# generate song_id-artist_name matrix
# code will run into RAM problem
songs_artist_long = songs[['song_id', 'artist_name']].astype(str)
songs_artist_long = pd.DataFrame(songs_artist_long.artist_name.str.split('|').
                                 tolist(),
                                 index=songs_artist_long.song_id).stack()
songs_artist_long = songs_artist_long.reset_index()[[0, 'song_id']]
songs_artist_long.columns = ['artist_name', 'song_id']
songs_artist_long['count'] = 1
songs_artist_wide = songs_artist_long.pivot(index='song_id',
                                            columns='artist_name',
                                            values='count')

# TODO: use gensim package to load and write songs table transformation with
# hard disk to bypass RAM issues

# Use gensim to deal with artist_name table transformation
t_artist_name = songs.artist_name.str.split('|').tolist()

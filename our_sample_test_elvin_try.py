import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard


# Load in the sample data with `read_csv()`
mysample = pd.read_csv("our_sample.txt", sep='\t')
print(mysample.dtypes)

# As you can see the data has 7 songs and 6 users
# Change the target column to 'category' type
mysample.target = mysample.target.astype('category')

# Tarek, my codes on song-artist_name won't work in your sample because the
# song-artist_name table come in the following format:
# song_id    artist_names
# song1      [artist1, artist2]
# song2      [artist2, artist3]
# song3      [artist1]
# The reason that I created a list of dictionary version (dict_artist) to
# apply "DictVectorizer" is because I need to flatten the list inside
# the artist_names list and put it into a Sparse Matrix. And DictVectorizer
# takes in dictionaries in below format:
# [{'artist1': 1, 'artist2': 1},
#  {'artist2': 1, 'artist3': 1},
#  {'artist1: 1'}]
# and it will return a scipy sparse matrix, with which I used TruncatedSVD
# to reduce dimensionality

# Therefore, in your sample data, msno is the corresponding 'song_id' while
# song_id is the corresponding 'artist_names' in my original codes. I have
# fixed it below for you. Please make sure you read my paragraph above to get a
# clear understanding of how it worked. Let me know if you have question!

# MATRIX FACTORIZATION ON MEMBER'S song_ids
# create list of unique identifier for later assembling (msno)
msno = mysample.msno.unique().tolist()
# generate dictionary for the sklearn DictVectorizer
dict_songs = [{name: 1 for name in mysample.loc[mysample.msno ==
              msno[i]].song_id.tolist()} for i in range(len(msno))]
print(dict_songs)
# Notice that dict_songs look like this:
# [{'Song1': 1, 'Song4': 1, 'Song2': 1, 'Song5': 1},
#  {'Song4': 1, 'Song5': 1},
#  {'Song5': 1},
#  {'Song6': 1, 'Song5': 1},
#  {'Song6': 1, 'Song7': 1},
#  {'Song1': 1, 'Song2': 1, 'Song3': 1}]
# each dictionary in this list corresponds to the name in the msno list:
# ['Tarek', 'John', 'Bob', 'David', 'Steve', 'Elvin']
# because the above code assemble it in this way

# Train a DictVectorizer based on dict_songs
vec_song = DictVectorizer().fit(dict_songs)
# Generate a CSR sparse matrix using the trained DictVectorizer
m_songs = vec_song.transform(dict_songs)
print(m_songs)  # this is a sparse matrix in CSR format
# Train a TruncatedSVD model with m_songs
svd_components = 4
rand_seed = 1122
svd_songs = TruncatedSVD(n_components=svd_components,
                         algorithm='randomized',
                         n_iter=5,
                         random_state=rand_seed).fit(m_songs)
# Generate a svd matrix for each member
v_songs = svd_songs.transform(m_songs)
print(v_songs)  # this should be a matrix of svd_components (4) columns
# Assemble a pd.DataFrame with the values from v_songs
songs_svds = ['song_id_svd_'+str(i+1) for i in range(svd_components)]
print(songs_svds)  # define column names
song_df = pd.DataFrame(v_songs,
                       columns=songs_svds)
song_df['msno'] = np.array(msno)
print(song_df)

"""
   song_id_svd_1  song_id_svd_2  song_id_svd_3  song_id_svd_4   msno
0       1.924155       0.297080      -0.199429      -0.307881  Tarek
1       1.077967      -0.575775      -0.565531      -0.253775   John
2       0.642089      -0.495815      -0.242788       0.414457    Bob
3       0.778192      -0.953606       0.432294       0.495555  David
4       0.159220      -0.641983       1.167777      -0.427391  Steve
5       0.989913       1.224048       0.633294       0.285143  Elvin
"""

# with this dataframe you can calculate the distance between rows

# Remember that we use this method because the code with pandas transformation
# will have the RAM full and get kill the process.
# Therefore for a small sample data we are working on here, it is almost
# guaranteed that your codes below will generate better distance values.
# the reason that I used DictVectorizer and TruncatedSVD is because we
# have to make a trade off between scale and performance
res = pdist(song_df.iloc[:, :-1], 'euclidean')
# since the SVDs are continuous, we need to switch out jaccard since it's
# almost certain that two floats won't be the same. also since output table
# is called distance, pdist already gives you the pair-wise distance
squareform(res)
distance = pd.DataFrame(squareform(res), index=song_df.msno,
                        columns=song_df.msno)
# I swapped out the df.index to df.msno since you probably need to see the
# results I suppose?
print(distance)
"""
msno      Tarek      John       Bob     David     Steve     Elvin
msno
Tarek  0.000000  1.270775  1.672132  1.980412  2.424947  1.666487
John   1.270775  0.000000  0.864338  1.337824  1.970527  2.230410
Bob    1.672132  0.864338  0.000000  0.830910  1.718409  1.965490
David  1.980412  1.337824  0.830910  0.000000  1.368576  2.207188
Steve  2.424947  1.970527  1.718409  1.368576  0.000000  2.228340
Elvin  1.666487  2.230410  1.965490  2.207188  2.228340  0.000000
"""
# So Tarek is close to John and Elvin, while Bob is similar to John and David
# It seems this method fits our instinct?

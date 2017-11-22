
# coding: utf-8

# In[21]:


# To make things clear i have created a very small readable sample data so we can test our matricies solution approach

import pandas as pd
# Load in the sample data with `read_csv()`
mysample = pd.read_csv("C:/our_sample.txt",sep='\t')
print(mysample)


# In[22]:


# As you can see the data has 7 songs and 6 users
#Change the target column to 'category' type
mysample.target=mysample.target.astype('category')


# In[23]:


# Now i will apply the code you provided on this sample (Consider that its user-song rather than artist-song "I didn't change the code for simplicity")
import numpy as np
import pandas as pd
#import gensim
#import standardized_import as stanimp
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD

# MATRIX FACTORIZATION ON SONG ARTIST NAMES
# create list of list for the multilabel variable
artist_names = mysample['msno']
# create list of unique identifier for later assembling
song_id = mysample.song_id.tolist()
# generate dictionary for the sklearn DictVectorizer
dict_artist = [{name: 1 for name in artist_names[i]} for
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
artist_df = pd.DataFrame(v_artist,
                         columns=artist_svds)
artist_df['song_id'] = np.array(song_id)


# In[25]:


artist_df


# In[28]:


# Now as you have seen in the artist_df above has 13 rows and 10 columns while we only have 6 users 

# Below is what i was trying to achieve using the same simple data

### Step 1  Transform dataframe from long to wide by Song_id

my_sample_wide=mysample.pivot_table(index='msno', columns='song_id',aggfunc=sum, fill_value=0)
flattened = pd.DataFrame(my_sample_wide.to_records())
flattened


# In[42]:


### Step 2  Create a distance (similarity) matrix using Jaccard 

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, jaccard

res = 1-pdist(flattened.iloc[:,1:], 'jaccard')
squareform(res)
distance = pd.DataFrame(squareform(res), index=flattened.index, columns= flattened.index)
distance


# In[ ]:


# As you can see in the 'distance' dataframe for example : Elvin and Tarek are having an 0.5 similarity as we have liked the same 2 songs 


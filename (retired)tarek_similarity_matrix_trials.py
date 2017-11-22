
# coding: utf-8

# In[10]:


# Import the `pandas` library as `pd`
import pandas as pd

# Import the `Numpy` library as `np`
import numpy as np

# Load in the train dataset with `read_csv()`
train = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/train.csv")

# Change target to Category data type
train['target']= train.target.astype('category')

# Drop unused columns from train dataset

model_train= train.drop(['source_system_tab','source_screen_name','source_type'], axis=1)


# In[7]:


# In order to generate a similarity matrix for distances, Data have to be in a wide format
# I will get the songs on wide format (Songs as columns users as our index rows) 
# So this will be our User-User similarity matrix
model_train_wide = model_train.pivot_table(index='msno', columns='song_id',aggfunc=sum, fill_value=0)


# In[12]:


# The method above runs out of memory (takes around 30 mins - 1 hour) , I found another method 
#using the sparse matrix method from Scipy (This ran overnight, i am sure more than 3 hours) however it ended with a weird errors
# i believe related to memory too

import pandas as pd
import numpy as np
#file=pd.read_csv("data.csv",names=['user','item','rating','timestamp'])

from scipy.sparse import csr_matrix

user_u = list(sorted(model_train.msno.unique()))
item_u = list(sorted(model_train.song_id.unique()))

row = model_train.msno.astype('category', categories=user_u).cat.codes
col = model_train.song_id.astype('category', categories=item_u).cat.codes

data = model_train['target'].tolist()

sparse_matrix = csr_matrix((data, (row, col)), shape=(len(user_u), len(item_u)))

df = pd.SparseDataFrame([ pd.SparseSeries(sparse_matrix[i].toarray().ravel() ,fill_value=0) 
                              for i in np.arange(sparse_matrix.shape[0]) ], 
                       index=user_u, columns=item_u, default_fill_value=0)




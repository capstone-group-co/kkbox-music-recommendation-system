
# coding: utf-8

# In[2]:


# Import the `pandas` library as `pd`
import pandas as pd

# Explore the Members Table

# Load in the data with `read_csv()`
members = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/members.csv")



# Name of columns
print(list(members))



# In[3]:


# Number of Rows and Columns
print(members.shape)


# In[5]:


# Summary Statistics of Columns
print(members.describe())


# In[6]:


# Number of Memebrs
print("{:,} Members".format(members.shape[0]))


# In[7]:


# Number of null values in each columns
print(members.isnull().sum())


# In[8]:


# Variable Types for Members Table
print(members.dtypes)


# In[9]:


###### ADJUST COLUMN TYPES

# Adjust Columns registration_init_time and expiration_date to datetime format

members["registration_init_time"] = members["registration_init_time"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
members["expiration_date"] = members["expiration_date"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

print(members.head)



# In[10]:


# Variable Types for `Members Table
print(members.dtypes)


# In[48]:


# Change City - Registred Via - Gender Columns into type Object
members['city'] = members['city'].astype('category')
members['registered_via'] = members['registered_via'].astype('category')
members['gender'] = members['gender'].astype('category')
print(members.dtypes)


# In[49]:


# Explore the Songs Table

# Load in the data with `read_csv()`
songs = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/songs.csv")



# In[50]:


# Name of columns
print(list(songs))

# Number of Rows and Columns
print("Rows","Columns",
    members.shape,)


# In[70]:


# Sample of Songs table
print(songs.head(3))


# In[52]:


# Variable Types for `Songs Table
print(songs.dtypes)


# In[53]:


print(songs.describe())


# In[54]:


#change Language to Object type without the decimals
import numpy as np
songs['language']=np.nan_to_num(songs['language']).astype(int)
songs['language']=songs.language.astype('category')


# In[55]:


# What are the Language codes available
songs.language.unique()


# In[57]:


# Variable Types for `Songs Table
print(songs.dtypes)


# In[60]:


# Change Song_Length to Float type
songs['song_length']= songs.song_length.astype(float)


# In[62]:


# Variable Types for `Songs Table
print(songs.dtypes)


# In[64]:


# Number of null values in each columns
print(songs.isnull().sum())


# In[66]:


# Explore the song_extra_info Table

# Load in the data with `read_csv()`
song_extra_info = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/song_extra_info.csv")


# In[68]:


# Name of columns
print(list(song_extra_info))

# Number of Rows and Columns
print("Rows","Columns",
    song_extra_info.shape,)


# In[71]:


# Variable Types for `Song_Extra_Info Table
print(song_extra_info.dtypes)


# In[74]:


# Sample of Song_Extra_Info table
print(song_extra_info.head(3))


# In[210]:


# Explore the train dataset

# Load in the train dataset with `read_csv()`
train = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/train.csv")


# In[211]:


# Name of columns
print(list(train))

# Number of Rows and Columns
print("Rows","Columns",
    train.shape,)

# Variable Types for train dataset
print(train.dtypes)


# In[212]:


# Change target to Category data type
train['target']= train.target.astype('category')


# In[213]:


# Sample of train dataset
print(train.head(3))


# In[214]:


# Distribution of training dataset for target 
train.target.value_counts()/len(train.target) *100


# In[215]:


# Number of null values in each columns
print(train.isnull().sum())


# In[216]:


# Number of Users in train dataset
print(len(train.msno.unique()),"Users in the train dataset") 


# In[263]:


######  Song Score Varibale  

#Create a Songs Score table copy of song id and target
songs_score= train[['song_id','target']].copy(deep=True)

#Rename Target column to Score
songs_score.columns=['song_id','score']

#Change Score type to float 
songs_score['score']=songs_score.score.astype(float)

#Replace all target 0 with -1 
songs_score.score=songs_score.score.replace(0,-1)

# Group by song id and sum scores column in order to have final score for each song
songs_score=songs_score.groupby('song_id',as_index=False).sum()


# In[274]:


# From this Score table we can check if the song is liked by many users or not (STILL UNDER CONSTRUCTION, I AM THINKING OF ALGORITHM) 
songs_score.describe()


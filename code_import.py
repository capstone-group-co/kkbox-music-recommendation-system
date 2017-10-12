
# coding: utf-8

# In[11]:


# Import the `pandas` library as `pd`
import pandas as pd

# Explore the Members Table

# Load in the data with `read_csv()`
members = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/members.csv")



# Name of columns
print(list(members))



# In[12]:


# Number of Rows and Columns
print(members.shape)


# In[13]:


# Summary Statistics of Columns
print(members.describe())


# In[14]:


# Number of Memebrs
print("{:,} Members".format(members.shape[0]))


# In[15]:


# Number of null values in each columns
print(members.isnull().sum())


# In[16]:


# Variable Types for Members Table
print(members.dtypes)


# In[17]:


###### ADJUST COLUMN TYPES

# Adjust Columns registration_init_time and expiration_date to datetime format

members["registration_init_time"] = members["registration_init_time"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
members["expiration_date"] = members["expiration_date"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

print(members.head)



# In[18]:


# Variable Types for `Members Table
print(members.dtypes)


# In[19]:


# Change City and Registred Via Columns into type Object
members['city'] = members['city'].astype(object)
members['registered_via'] = members['registered_via'].astype(object)
print(members.dtypes)


# In[20]:


import matplotlib.pyplot as plt

plt.hist(members['city'],bins=21)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


# In[72]:


# Explore the Songs Table

# Load in the data with `read_csv()`
songs = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/songs.csv")



# In[73]:


# Name of columns
print(list(songs))

# Number of Rows and Columns
print("Rows","Columns",
    members.shape,)


# In[74]:


# Sample of Songs table
print(songs.head())


# In[82]:


# Variable Types for `Songs Table
print(songs.dtypes)


# In[81]:


print(songs.describe())


# In[117]:


# TRYING TO CHANGE THE SONG_LENGTH TO Minutes and Seconds (Still under process)

import numpy as np
sec=np.array(songs["song_length"])
sec=sec/60
sec=sec/60
print(sec.mean())
print(sec.max())



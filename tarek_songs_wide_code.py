
# coding: utf-8

# In[1]:


# Before going into the Collaborative Filtering steps, i thought i would need to convert the songs and genre from long to wide format,
# This will be used later on with the Content Based Algorithm

#the following 2 steps for creating a wide format song with a unique genre columns. 

# The next part in the begining i thought it's a must to determine     

#############################Create a wide dataframe for songs genre_ids 

#### Data Required :   For this purpose i only need Song_id and Genre_ids from the Songs table

# Import the `pandas` library as `pd`
import pandas as pd

# Load in the Songs data with `read_csv()`
songs = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/songs.csv")


songs_wide = songs.iloc[:,[0,2]]

#### Change genre_ids to type string

songs_wide['genre_ids']=songs_wide['genre_ids'].astype(str)

#print(songs_wide)

### Step 1  For songs having more than one genre_ids seperate into multiple rows

songs_wide=pd.DataFrame(songs_wide.genre_ids.str.split('|').tolist(), index=songs_wide.song_id).stack()
songs_wide = songs_wide.reset_index()[[0, 'song_id']] # genre_ids variable is currently labeled 0
songs_wide.columns = ['genre_ids', 'song_id'] # renaming genre_ids


### Now its clear we have 192 unique Genres including one for NA

len(songs_wide.genre_ids.unique())

### Step 2  Transform dataframe from long to wide by genre_id

songs_wide= songs_wide.pivot_table(index='song_id', columns='genre_ids', 
                        aggfunc=len, fill_value=0)


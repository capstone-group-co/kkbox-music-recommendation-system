
# coding: utf-8

# In[54]:


# Import the `Train_Test Split function from Scikit library` library
from sklearn.model_selection import train_test_split

# Import the `pandas` library as `pd`
import pandas as pd

# Import the `Numpy` library as `np`
import numpy as np

# Load in the train dataset with `read_csv()`
train = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/train.csv")

# Change target to Category data type
train['target']= train.target.astype('category')


# In[55]:


# Divide the train dataset to train and test 
model_train, model_test = train_test_split(train, test_size=0.2)


# In[57]:


print(model_train.shape,model_test.shape)


# In[58]:


# Use 'Surprise' library for building a CF(Collaborative Filtering) Recommendation System

### First, 'Surprise' library input requires only User_id, Song_id and Target   

# Drop unused columns from model_train and model_test

model_train= model_train.drop(['source_system_tab','source_screen_name','source_type'], axis=1)

model_test= model_test.drop(['source_system_tab','source_screen_name','source_type'], axis=1)


# In[60]:


from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 1))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(model_train[['msno', 'song_id', 'target']], reader)


# In[61]:


#select the SVD algorithm from the Surprise library 
algo = SVD()

# Transorm the data to Surprise datatype 
trainset = data.build_full_trainset()

#Train the model
algo.train(trainset)


# In[68]:


# Create a for loop to predict each row of the splitted test and save it to my_pred dataframe
my_pred=[]
for i in (list(range(len(model_test)))):
    #print(i)
    userid = str(model_test.msno.iloc[i])
    itemid = str(model_test.song_id.iloc[i])
    actual_rating = model_test.target.iloc[i]
    pred=algo.predict(userid,itemid,actual_rating)
    predicted_rating=pred.est
    my_pred.append([userid,itemid,actual_rating,predicted_rating])
    
my_pred=pd.DataFrame(my_pred)
my_pred.columns=['userid','itemid','actual_rating','predicted_rating']
my_pred.sample(10)    


# In[ ]:


# Results looking good, we need to create a threshold  for example "IF predicted_rating > 0.5 then '1' else '0')   

# once we get a good threshold we apply it on the test dataset and submit and check our rankings.

# You can use this predicted_rating as an additional varibale to predict 


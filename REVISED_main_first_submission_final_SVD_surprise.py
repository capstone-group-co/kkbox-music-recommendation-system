"""
Script to create prediction based purely on msno-song_id combination using
collaborative filtering algorithm with Surprise package
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import SVD
from surprise import Dataset
# from surprise import evaluate, print_perf
from surprise import Reader

train = pd.read_csv("../raw_data/train.csv")
train['target'] = train.target.astype('category')

# Divide the train dataset to train and test
model_train, model_test = train_test_split(train, test_size=0.2)
print(model_train.shape, model_test.shape)

# Use 'Surprise' library for building a CF(Collaborative Filtering)
# Recommendation System
# First, 'Surprise' library input requires only User_id, Song_id and Target

# Drop unused columns from model_train and model_test
model_train = model_train.drop(
    ['source_system_tab', 'source_screen_name', 'source_type'], axis=1)

model_test = model_test.drop(
    ['source_system_tab', 'source_screen_name', 'source_type'], axis=1)
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 1))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(model_train[['msno', 'song_id', 'target']], reader)

# select the SVD algorithm from the Surprise library
algo = SVD()
# Transorm the data to Surprise datatype
trainset = data.build_full_trainset()
# Train the model
algo.train(trainset)

# Create a for loop to predict each row of the splitted test and save it
# to my_pred dataframe
my_pred = []
for i in (list(range(len(model_test)))):
    # print(i)
    userid = str(model_test.msno.iloc[i])
    itemid = str(model_test.song_id.iloc[i])
    actual_rating = model_test.target.iloc[i]
    pred = algo.predict(userid, itemid, actual_rating)
    predicted_rating = pred.est
    my_pred.append([userid, itemid, actual_rating, predicted_rating])

my_pred = pd.DataFrame(my_pred)
my_pred.columns = ['userid', 'itemid', 'actual_rating', 'predicted_rating']
print(my_pred.sample(10))


# Now apply the model on the competition test dataset to submit our first trial
# Load in the test dataset with `read_csv()`
test = pd.read_csv("C:/Data/Fall 2017/Capstone/raw_data/test.csv")
# Create a for loop to predict each row of the final test and save it to
# final_pred dataframe
final_pred = []
for i in (list(range(len(test)))):
    test_id = str(test.id.iloc[i])
    userid = str(test.msno.iloc[i])
    itemid = str(test.song_id.iloc[i])
    pred = algo.predict(userid, itemid, actual_rating)
    predicted_rating = pred.est
    final_pred.append([test_id, predicted_rating])

final_pred = pd.DataFrame(final_pred)
final_pred.columns = ['id', 'target']


# Save the final_pred to CSV to submit
final_pred.to_csv('first_submission.csv', index=False)

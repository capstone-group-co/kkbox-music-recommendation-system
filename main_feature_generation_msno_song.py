import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader

# create train dataframe
train = pd.read_csv("../raw_data/train.csv")
train['target'] = train.target.astype('category')

# create training data for cf_algo
train_cf = train.drop(
    ['source_system_tab', 'source_screen_name', 'source_type'], axis=1)


def train_cf_algo(model_data):
    print(">>> training cf model...")
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(
        model_data[['msno', 'song_id', 'target']], reader)
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.train(trainset)
    return algo


def get_cf_score(model_data, trained_algo):
    print(">>> generating cf predicted scores...")
    score = []
    for i in range(len(model_data)):
        userid = str(model_data.msno.iloc[i])
        itemid = str(model_data.song_id.iloc[i])
        pred = trained_algo.predict(userid, itemid)
        predicted_rating = pred.est
        score.append([userid, itemid, predicted_rating])
    score = np.array(score)
    return(score)

# train algo based on the full training data
cf_algo = train_cf_algo(train_cf)

# predict cf_score for the full training data
train['cf_score'] = get_cf_score(train_cf, cf_algo)
print('>>> cf score generated to train dataframe')
print(train.head())

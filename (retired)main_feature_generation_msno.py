import numpy as np
import pandas as pd
import datetime
import main_standardized_import as stanimp
from sklearn.preprocessing import LabelBinarizer

file_names = ['../raw_data/members.csv',
              '../raw_data/song_extra_info.csv',
              '../raw_data/songs.csv',
              '../raw_data/test.csv',
              '../raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

members, song_extra_info, songs, test, train =\
    stanimp.kkbox_cleaning(members, song_extra_info, songs, test, train)

del songs, song_extra_info, test, train

print(">>> start feature generation for members table")
print(">>> start binarize members gender")
# binarize language categories
gender_lb = LabelBinarizer()
gender_lb.fit(members.gender.values.astype(str).reshape(-1, 1))
gender_bi_v = gender_lb.transform(
    members.gender.values.astype(str).reshape(-1, 1))
gender_binaries = ['gender_bi_'+str(i+1) for i in range(gender_bi_v.shape[1])]
for i, name in enumerate(gender_binaries):
    members[name] = gender_bi_v[:, i].astype(float)

print(">>> start binarize members city")
# binarize language categories
city_lb = LabelBinarizer()
city_lb.fit(members.city.values.astype(int).reshape(-1, 1))
city_bi_v = city_lb.transform(
    members.city.values.astype(int).reshape(-1, 1))
city_binaries = ['city_bi_'+str(i+1) for i in range(city_bi_v.shape[1])]
for i, name in enumerate(city_binaries):
    members[name] = city_bi_v[:, i].astype(float)

print(">>> start binarize members registered_via")
# binarize language categories
rv_lb = LabelBinarizer()
rv_lb.fit(members.registered_via.values.astype(int).reshape(-1, 1))
rv_bi_v = rv_lb.transform(
    members.registered_via.values.astype(int).reshape(-1, 1))
rv_binaries = ['rv_bi_'+str(i+1) for i in range(rv_bi_v.shape[1])]
for i, name in enumerate(rv_binaries):
    members[name] = rv_bi_v[:, i].astype(float)

print(">>> generating day_since_founding for membership init time")
# create day_since_founding
launch_time = datetime.datetime.strptime('2004-10-01', '%Y-%m-%d')
members['day_since_founding'] = ((
    members.registration_init_time -
    launch_time)/np.timedelta64(1, 'D')).astype(float)

print(">>> generating day_membership_duration before membership expires")
# create day_membership_duration
members['day_membership_duration'] = ((
    members.expiration_date - members.registration_init_time) / np.timedelta64(
    1, 'D')).astype(float)

print(">>> removing unwanted columns")
# delete categorical variables and datetime variables
members = members.drop(['city', 'gender', 'registered_via',
                        'registration_init_time', 'expiration_date'], axis=1)
print(members.head())

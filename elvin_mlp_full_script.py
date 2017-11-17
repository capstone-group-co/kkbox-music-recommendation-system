import numpy as np
import pandas as pd
import datetime
import main_standardized_import as stanimp

from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, Imputer
from sklearn.metrics import confusion_matrix

from surprise import SVD
from surprise import Dataset
from surprise import Reader

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

output_file = 'submission.csv'
file_names = ['../raw_data/members.csv',
              '../raw_data/song_extra_info.csv',
              '../raw_data/songs.csv',
              '../raw_data/test.csv',
              '../raw_data/train.csv']

members, song_extra_info, songs, test, train = [pd.read_csv(x)
                                                for x in file_names]

members, song_extra_info, songs, test, train =\
    stanimp.kkbox_cleaning(members, song_extra_info, songs, test, train)

"""
SONGS TABLE FEATURE ENGINEERING
"""
print(">>> SONGS TABLE FEATURE ENGINEERING")
print(">>> start binarizing songs language")
# binarize language categories
language_lb = LabelBinarizer()
language_lb.fit(songs.language.values.astype(int).reshape(-1, 1))
lang_bi_v = language_lb.transform(
    songs.language.values.astype(int).reshape(-1, 1))
lang_binaries = ['language_bi_'+str(i+1) for i in range(lang_bi_v.shape[1])]
for i, name in enumerate(lang_binaries):
    songs[name] = lang_bi_v[:, i]

# matrix factorization on song artist_name
print(">>> start matrix factorization for songs artist_name")
artist_names = songs['artist_name'].str.split('&').tolist()
song_id = songs.song_id.tolist()
dict_artist = [{name.strip(): 1 for name in artist_names[i]} for
               i in range(len(song_id))]
vec_artist = DictVectorizer().fit(dict_artist)
m_artist = vec_artist.transform(dict_artist)
svd_components = 10
rand_seed = 1122
svd_artist_name = TruncatedSVD(n_components=svd_components,
                               algorithm='randomized',
                               n_iter=5,
                               random_state=rand_seed).fit(m_artist)
v_artist = svd_artist_name.transform(m_artist)
artist_svds = ['artist_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(artist_svds):
    songs[name] = v_artist[:, i]

# matrix factorization on genre_ids
print(">>> start matrix factorization for songs genre_ids")
genre_ids = songs['genre_ids'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_genre = [{} if not isinstance(genre_ids[i], list)
              else {genre_id.strip(): 1 for genre_id in genre_ids[i]}
              for i in range(len(genre_ids))]
vec_genre = DictVectorizer().fit(dict_genre)
m_genre = vec_genre.transform(dict_genre)
svd_components = 10
rand_seed = 1122
svd_genre_id = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_genre)
v_genre = svd_genre_id.transform(m_genre)
genre_svds = ['genre_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(genre_svds):
    songs[name] = v_genre[:, i]

# matrix factorization on song composer
print(">>> start matrix factorization for songs composer")
composers = songs['composer'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_composer = [{} if not isinstance(composers[i], list)
                 else {composer.strip(): 1 for composer in composers[i]}
                 for i in range(len(composers))]
vec_composer = DictVectorizer().fit(dict_composer)
m_composer = vec_composer.transform(dict_composer)
svd_components = 10
rand_seed = 1122
svd_composer = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_composer)
v_composer = svd_composer.transform(m_composer)
composer_svds = ['composer_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(composer_svds):
    songs[name] = v_composer[:, i]

# matrix factorization on lyricist
print(">>> start matrix factorization for songs lyricist")
lyricists = songs['lyricist'].str.split('|').tolist()
song_id = songs.song_id.tolist()
dict_lyricist = [{} if not isinstance(lyricists[i], list)
                 else {lyricist.strip(): 1 for lyricist in lyricists[i]}
                 for i in range(len(lyricists))]
vec_lyricist = DictVectorizer().fit(dict_lyricist)
m_lyricist = vec_lyricist.transform(dict_lyricist)
svd_components = 10
rand_seed = 1122
svd_lyricist = TruncatedSVD(n_components=svd_components,
                            algorithm='randomized',
                            n_iter=5,
                            random_state=rand_seed).fit(m_lyricist)
v_lyricist = svd_lyricist.transform(m_lyricist)
lyricist_svds = ['lyricist_svd_'+str(i+1) for i in range(svd_components)]
for i, name in enumerate(lyricist_svds):
    songs[name] = v_lyricist[:, i]

print(">>> removing unwanted columns")
songs = songs.drop(
    ['genre_ids', 'artist_name', 'composer', 'lyricist', 'language'],
    axis=1)
print(">>> songs table feature engineering completed")

"""
MEMBERS TABLE FEATURE ENGINEERING
"""
print(">>> MEMBERS TABLE FEATURE ENGINEERING")
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
print(">>> members table feature generation completed")

"""
MEMBER-SONG TABLE FEATURE ENGINEERING
"""
print(">>> MEMBER-SONG FEATURE ENGINEERING")
print(">>> start cleaning train and test")


def main_table_engineering(main_df):
    # binarize source_system_tab
    system_tab_lb = LabelBinarizer()
    system_tab_lb.fit(main_df.source_system_tab.values.astype(
        str).reshape(-1, 1))
    system_tab_bi_v = system_tab_lb.transform(
        main_df.source_system_tab.values.astype(str).reshape(-1, 1))
    system_tab_binaries = ['system_tab_bi_'+str(i+1) for i in
                           range(system_tab_bi_v.shape[1])]
    for i, name in enumerate(system_tab_binaries):
        main_df[name] = system_tab_bi_v[:, i].astype(float)
    # binarize source_screen_name
    screen_name_lb = LabelBinarizer()
    screen_name_lb.fit(main_df.source_screen_name.values.astype(
        str).reshape(-1, 1))
    screen_name_bi_v = screen_name_lb.transform(
        main_df.source_screen_name.values.astype(str).reshape(-1, 1))
    screen_name_binaries = ['screen_name_bi_'+str(i+1) for i in
                            range(screen_name_bi_v.shape[1])]
    for i, name in enumerate(screen_name_binaries):
        main_df[name] = screen_name_bi_v[:, i].astype(float)
    # binarize source_type
    source_type_lb = LabelBinarizer()
    source_type_lb.fit(main_df.source_type.values.astype(
        str).reshape(-1, 1))
    source_type_bi_v = source_type_lb.transform(
        main_df.source_type.values.astype(str).reshape(-1, 1))
    source_type_binaries = ['source_type_bi_'+str(i+1) for i in
                            range(source_type_bi_v.shape[1])]
    for i, name in enumerate(source_type_binaries):
        main_df[name] = source_type_bi_v[:, i].astype(float)
    main_df = main_df.drop(
        ['source_system_tab', 'source_screen_name', 'source_type'], axis=1)
    return main_df

temp_df = train.drop(['target'], axis=1).append(
    test.drop(['id'], axis=1))
temp_df = main_table_engineering(temp_df)

train = pd.concat([train, temp_df.iloc[:train.shape[0], 2:]], axis=1)
test = pd.concat([test, temp_df.iloc[train.shape[0]:, 2:]], axis=1)
print(">>> train and test set cleaned")
del temp_df

"""
COLLABORATIVE FILTERING SCORE GENERATION
"""
print(">>> training collaborative filtering system for cf scoring")
train['target'] = train.target.astype('category')

# create separate training data for cf_algo
train_cf = train[['msno', 'song_id', 'target']]


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
        score.append(predicted_rating)
    score = np.array(score)
    return(score)

# train algo based on the full training data
cf_algo = train_cf_algo(train_cf)

# predict cf_score for the full training data
print(">>> generating cf score for train")
train['cf_score'] = get_cf_score(train_cf, cf_algo)
print(">>> generating cf score for test")
test['cf_score'] = get_cf_score(test, cf_algo)
print(">>> cf score generated")

del train_cf
print(train.columns.tolist())
print(test.columns.tolist())

"""
FINAL TABLE ASSEMBLING
"""
train = train.drop(['source_system_tab', 'source_screen_name', 'source_type'],
                   axis=1)
test = test.drop(['source_system_tab', 'source_screen_name', 'source_type'],
                 axis=1)

print(">>> merging train with songs and members")
train = train.merge(songs, on='song_id').merge(members, on='msno')
print(">>> train now has %i variables and %i observations" %
      (train.shape[1], train.shape[0]))
print(">>> merging test with songs and members")
test = test.merge(songs, on='song_id').merge(members, on='msno')
print(">>> test now has %i variables and %i observations" %
      (test.shape[1], test.shape[0]))
print(train.columns.tolist())
print(test.columns.tolist())
print(">>> removing unwanted tables to save RAM")
del songs, members, song_extra_info

# training MLP
# set random seed
torch.manual_seed(1122)


print(">>> adapt train and test for pytorch training")
train = train.iloc[:, 2:].as_matrix().astype(float)
test_id = test.id.astype(float).values
test = test.iloc[:, 3:].as_matrix().astype(float)

print(">>> generate train, val, test data for mlp")
# create np matrices
x_train, x_val, y_train, y_val = train_test_split(
    train[:, 1:], train[:, 0], test_size=0.001)
x_test, y_test = test[:, :], np.zeros((test.shape[0], 1))

# impute missing values
print(">>> impute missing values")
imputer = Imputer()
x_train = imputer.fit_transform(x_train)
x_val = imputer.transform(x_val)
x_test = imputer.transform(x_test)

# rescale training data
print(">>> rescale data")
x_train = scale(x_train, axis=0)
x_val = scale(x_val, axis=0)
x_test = scale(x_test, axis=0)

# create pytorch compatible dataset that has API for automated loaders
trainset = TensorDataset(torch.Tensor(x_train.tolist()).view(
                         x_train.shape[0], -1),
                         torch.Tensor(y_train.tolist()).long())

valset = TensorDataset(torch.Tensor(x_val.tolist()).view(
                       x_val.shape[0], -1),
                       torch.Tensor(y_val.tolist()).long())

testset = TensorDataset(torch.Tensor(x_test.tolist()).view(
                        x_test.shape[0], -1),
                        torch.Tensor(y_test.tolist()).long())

# create pytorch mini-batch loader DataLoader for the dataset
trainloader = DataLoader(trainset, batch_size=2000, shuffle=True)

valloader = DataLoader(valset, batch_size=2000, shuffle=True)

# for test set, we want to maintain the sequence of the data
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=2000, shuffle=False,
                        sampler=testsampler)
print(">>> train, val, test dataset created")


# define and initialize a multilayer-perceptron, a criterion, and an optimizer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(126, 20)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(20, 2)
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x

print(">>> initiate mlp models")
mlp = MLP()
mlp.cuda()
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validateModel(dataloader, epoch):
    mlp.eval()
    test_loss = 0
    correct = 0
    pred = np.array([])
    targ = np.array([])
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = mlp(inputs)
        test_loss += F.nll_loss(outputs, targets, size_average=False).data[0]
        pred = np.append(pred, outputs.topk(1)[1].data.view(1, -1).numpy())
        targ = np.append(targ, targets.data.numpy())
        prd = outputs.topk(1)[1].data
        correct += prd.eq(targets.data.view_as(prd)).cpu().sum()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    cm = confusion_matrix(targ, pred)
    print('[Epoch %i] Accuracy: %.2f, Average Loss: %.2f' %
          (epoch, test_acc, test_loss))
    print(cm)
    return test_loss, test_acc, cm


def testModel(dataloader):
    mlp.eval()
    pred = np.array([])
    for inputs, _ in dataloader:
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred,
                         outputs.topk(1)[1].data.view(1, -1).cpu().numpy())
    return pred

# run the training epoch 100 times and test the result
print(">>> training model with mlp")
epoch_loss = []
epoch_acc = []
for epoch in range(30):
    trainEpoch(trainloader, epoch)
    loss, acc, _ = validateModel(valloader, epoch)
    epoch_loss.append(loss)
    epoch_acc.append(acc)

print(">>> creating predictions with mlp")
pred = 1 - testModel(testloader)

print(">>> outputing predictions to local file")
pred = pred.astype(int)
submission = pd.DataFrame()
submission['id'] = test_id.astype(int) + 1
submission['target'] = pred
submission.to_csv('submission.csv', index=False)

epoch_performance = pd.DataFrame()
epoch_performance['epoch_id'] = np.arange(len(epoch_loss)) + 1
epoch_performance['epoch_loss'] = np.array(epoch_loss)
epoch_performance['epoch_acc'] = np.array(epoch_acc)
epoch_performance.to_csv('epoch_performance.csv', index=False)

# TODO: fix memory error problem by deleting objects unused

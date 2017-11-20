import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

output_file = 'test_submission.csv'
sqlite_url = 'sqlite:///kkbox.db'
train_table = 'full_train'
test_table = 'full_test'
batch_size = 5000
'''
b: 5000, loading: 8, training: 0, testing: 3.5
b: 20000, loading: 30, training: 0, testing: 14
b: 80000, loading: 100, training: 0.1, testing: 54 --best batch_size
b: 100000, loading: 123, training: 0.1, testing: 67
'''

# set random seed
torch.manual_seed(1122)

# create linkage to database
print(">>> link to sqlite database kkbox.db")
kkbox_engine = create_engine('sqlite:///kkbox.db')
kkbox_conn = kkbox_engine.connect()


class KKBOXDataset(Dataset):
    """KKBOX Final Set (SQLITE) for Training and Testing."""
    def __init__(self, sql_conn, table_name):
        """
        Args:
        sql_url: dabase url used for sqlalchemy to build engine
        table_name: respective table to query for
        """
        self.sql_conn = sql_conn
        self.table_name = table_name

    def __len__(self):
        ln = self.sql_conn.execute(
            'SELECT max(rowid) FROM ' + self.table_name).scalar()
        return ln

    def __getitem__(self, idx):
        stmt = 'SELECT * FROM ' + self.table_name + ' WHERE rowid = ' + str(
            idx + 1)
        line = self.sql_conn.execute(stmt).fetchone()
        # print('fetched:', line[-1])
        line = self.make_list(line)
        line = line[1:-1]
        # print('sliced:', line)
        line = np.asarray(line, dtype=float)
        return line

    def make_list(self, line):
        return [x for x in line]


trainset = KKBOXDataset(kkbox_conn, train_table)
testset = KKBOXDataset(kkbox_conn, test_table)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        sampler=testsampler)
print(">>> train, test dataset created")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(131, 20)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(20, 2)
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x


print(">>> initiate mlp models")
mlp = MLP()
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    start = time.time()
    for i, data in enumerate(dataloader, 0):
        print("Epoch %i, Iter %i:" % (epoch + 1, i + 1))
        inputs, labels = data[:, 1:], data[:, 0]
        end = time.time()
        print("Loading one batch takes %.4f seconds" % (end - start))
        inputs, labels = inputs.float(), labels.long()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break


def testModel(dataloader):
    mlp.eval()
    pred = np.array([])
    for data in dataloader:
        inputs = data[:, 1:].float()
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred,
                         outputs.topk(1)[1].data.view(1, -1).numpy())
        pred = 1 - pred
        break
    return pred


# run the training epoch 100 times and test the result
print(">>> training model with mlp")
train_start = time.time()
trainEpoch(trainloader, 0)
train_end = time.time()
print(">>> training one batch takes %.4f seconds" % (train_end - train_start))

print(">>> creating predictions with mlp")
test_start = time.time()
pred = testModel(testloader)
test_end = time.time()
print(">>> testing one batch takes %.4f seconds" % (test_end - test_start))

print(">>> outputing batch predictions to local file")
pred = pred.astype(int)
stmt3 = "SELECT id FROM full_test"
test_id = kkbox_conn.execute(stmt3).fetchall()
test_id = np.array([x[0] for x in test_id]).astype(int)[:batch_size]
submission = pd.DataFrame()
submission['id'] = test_id
submission['target'] = pred
submission.to_csv(output_file, index=False)

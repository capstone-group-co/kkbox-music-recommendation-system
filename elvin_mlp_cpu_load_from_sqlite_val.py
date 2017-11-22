import numpy as np
import pandas as pd
import datetime
from sqlalchemy import create_engine

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

output_file = 'submission.csv'
save_model = 'mlp_model' + datetime.datetime.now().strftime(
    "%Y_%m_%d_%H_%M") + ".pt"
sqlite_url = 'sqlite:///kkbox.db'
train_table = 'full_train'
test_table = 'full_test'
batch_size = 5000
total_epochs = 1
num_workers = 0
val_freq = 5
learning_rate = 1
momentum = 0.7

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
                         shuffle=True, num_workers=num_workers)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        sampler=testsampler, num_workers=num_workers)
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
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=momentum)


# define a training epoch function
def trainEpoch(dataloader, epoch, val_freq=20):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    for i, data in enumerate(dataloader, 0):
        print("Epoch %i, Iter %i" % (epoch + 1, i + 1))
        inputs, labels = data[:, 1:], data[:, 0]
        inputs, labels = inputs.float(), labels.long()
        inputs, labels = Variable(inputs), Variable(labels)
        if (i + 1) % val_freq == 0:
            mlp.eval()
            outputs = mlp(inputs)
            loss = F.nll_loss(outputs, labels).data[0]
            prd = 1 - outputs.topk(1)[1].data
            correct = prd.eq(labels.data.view_as(prd)).sum()
            acc = correct / len(dataloader.dataset)
            print("Accuracy: %.2f percent" % (acc * 100))
            mlp.train()
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


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
    return pred


# run the training epoch 100 times and test the result
print(">>> training model with mlp")
for epoch in range(total_epochs):
    trainEpoch(trainloader, epoch, val_freq)

print(">>> creating predictions with mlp")
pred = testModel(testloader)

print(">>> saving model to local path")
torch.save(mlp.state_dict(), save_model)

print(">>> outputing predictions to local file")
pred = pred.astype(int)
stmt3 = "SELECT id FROM full_test"
test_id = kkbox_conn.execute(stmt3).fetchall()
test_id = np.array([x[0] for x in test_id]).astype(int)
submission = pd.DataFrame()
submission['id'] = test_id
submission['target'] = pred
submission.to_csv(output_file, index=False)
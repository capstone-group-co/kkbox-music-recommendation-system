"""
Script to load pre-trained model and output
predictions
"""

import numpy as np
import pandas as pd
import datetime
from sqlalchemy import create_engine

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
# from torchnet.meter import AUCMeter

sqlite_url = 'sqlite:///../kkbox.db'
test_table = 'full_test'
batch_size = 250
num_workers = 0
load_model = '../logs/mlp_model_2017_11_22_19_10_epoch_1_lr_0.1_m_0.9.pt'
last_epoch = 1

# set random seed
torch.manual_seed(1122)

# create linkage to database
print(">>> link to sqlite database kkbox.db")
kkbox_engine = create_engine(sqlite_url)
kkbox_conn = kkbox_engine.connect()


class KKBOXDataset(Dataset):
    """KKBOX Final Set (SQLITE) for Training and Testing."""
    def __init__(self, sql_conn, table_name):
        """
        Args:
        sql_conn: dabase connection used by sqlalchemy to build engine
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
        line = [x for x in line]
        line = line[1:]
        # print('sliced:', line)
        line = np.asarray(line, dtype=float)
        line[np.isnan(line)] = 0
        return line


testset = KKBOXDataset(kkbox_conn, test_table)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        sampler=testsampler, num_workers=num_workers)
print(">>> test dataset created")


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(130, 150)
        self.t1 = nn.ReLU()
        self.d1 = nn.Dropout(p=0.6)
        self.l2 = nn.Linear(150, 50)
        self.t2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(50, 2)
        self.t3 = nn.LogSoftmax()

    def forward(self, x):
        x = self.d1(self.t1(self.l1(x)))
        x = self.d2(self.t2(self.l2(x)))
        x = self.t3(self.l3(x))
        return x


print(">>> initiate mlp models")
mlp = MLP()
# load local model if specified
if load_model:
    mlp.load_state_dict(torch.load(load_model))


def testModel(dataloader, epoch):
    print("Testing Epoch %i" % (epoch + 1))
    mlp.eval()
    pred = np.array([])
    for i, data in enumerate(dataloader):
        if i % 100 == 0:
            print("Generating output batch #%i" % (i + 1))
            print("--%.2f percent completed---" % (100 * i/len(
                dataloader.dataset)))
        inputs = data[:, 1:].float()
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred, np.exp(outputs.data.select((1), 1).contiguous(
            ).view(1, -1).numpy()))
    return pred


# define test_id for submission
stmt = "SELECT id FROM full_test"
test_id = kkbox_conn.execute(stmt).fetchall()
test_id = np.array([x[0] for x in test_id]).astype(int)

# create output with trained model
print(">>> creating predictions with mlp")
pred = testModel(testloader, 0)
print(">>> outputing predictions to local file")
pred = pred.astype(float)
submission = pd.DataFrame()
submission['id'] = test_id
submission['target'] = pred
output_file = 'submission_' + datetime.datetime.now().strftime(
    "%Y_%m_%d_%H_%M") + '_epoch_' + str(last_epoch) + ".csv.gz"
submission.to_csv(output_file, compression='gzip', index=False,
                  float_format='%.5f')

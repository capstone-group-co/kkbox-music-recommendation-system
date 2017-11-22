import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch

sqlite_url = 'sqlite:///../kkbox.db'
train_table = 'full_train'
test_table = 'full_test'

# create linkage to database
print(">>> link to sqlite database kkbox.db")
kkbox_engine = create_engine('sqlite:///kkbox.db')
kkbox_conn = kkbox_engine.connect()


# reflect table structures with Table object
metadata = MetaData(kkbox_engine)
full_train = Table('full_train', metadata, autoload=True,
                   autoload_with=kkbox_engine)
full_test = Table('full_test', metadata, autoload=True,
                  autoload_with=kkbox_engine)
print(full_train.c.keys())
print(full_test.c.keys())


def get_record(command, conn):
    r = conn.execute(command).fetchall()
    r = [x for x in r]
    r = np.asarray(r, dtype=float)
    return r

command = "SELECT target FROM full_train"
x = get_record(command, kkbox_conn)
print(x)

command = 'SELECT * FROM full_train WHERE rowid = 1'
y = get_record(command, kkbox_conn)
print(y)

command = 'SELECT max(rowid) FROM full_train'
z = get_record(command, kkbox_conn)
print(z)


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
        l = self.sql_conn.execute(
            'SELECT max(rowid) FROM ' + self.table_name).scalar()
        return l

    def __getitem__(self, idx):
        stmt = 'SELECT * FROM ' + self.table_name + ' WHERE rowid = ' + str(
            idx + 1)
        line = self.sql_conn.execute(stmt).fetchone()
        print('fetched:', line[-1])
        line = self.make_list(line)
        line = line[1:-1]
        print('sliced:', line)
        line = np.asarray(line, dtype=float)
        return line

    def make_list(self, line):
        return [x for x in line]

trainset = KKBOXDataset(kkbox_conn, train_table)
testset = KKBOXDataset(kkbox_conn, test_table)
trainloader = DataLoader(trainset, batch_size=250, shuffle=True)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=250, shuffle=False,
                        sampler=testsampler)

for i in trainloader:
    x = i
    break
# select all id from full_test
stmt3 = "SELECT id FROM full_test"
test_id = kkbox_conn.execute(stmt3).fetchall()
test_id = [x[0] for x in test_id]
test_id = np.array(test_id)
test_id = torch.from_numpy(test_id)

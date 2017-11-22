import numpy as np
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch

output_file = 'submission.csv'
sqlite_url = 'sqlite:///kkbox.db'
train_table = 'full_train'
test_table = 'full_test'

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

stmt3 = "SELECT id FROM full_test"
test_id = kkbox_conn.execute(stmt3).fetchall()
test_id = [x[0] for x in test_id]
test_id = np.array(test_id)
test_id = torch.from_numpy(test_id)

test_id = torch.from_numpy(test_id[:100])
x = torch.FloatTensor()
y = torch.LongTensor()
'''
# reflect table structures with Table object
from sqlalchemy import MetaData, Table
metadata = MetaData(kkbox_engine)
full_train = Table('full_train', metadata, autoload=True,
                   autoload_with=kkbox_engine)
full_test = Table('full_test', metadata, autoload=True,
                   autoload_with=kkbox_engine)


def get_record(command, conn):
    r = conn.execute(command).fetchall()
    r = [x for x in r]
    r = np.asarray(r, dtype=float)
    return r

command = "SELECT target FROM full_train"
x = get_record(command, kkbox_conn)

stmt1 = 'SELECT * FROM full_train WHERE rowid = 1'
result1 = kkbox_conn.execute(stmt1).fetchone()
result1 = [x for x in result1]
len(result1[1:-1])
np.asarray(result1, dtype=float)

stmt2 = 'SELECT max(rowid) FROM full_train'
result2 = kkbox_conn.execute(stmt2).scalar()
result2
'''

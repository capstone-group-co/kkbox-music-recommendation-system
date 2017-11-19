import numpy as np
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

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
trainsampler = RandomSampler(trainset)
trainloader = DataLoader(trainset, batch_size=2, shuffle=True)
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=2, shuffle=False,
                        sampler=testsampler)


# reflect table structures with Table object
# metadata = MetaData(kkbox_engine)
# full_train = Table('full_train', metadata, autoload=True,
#                   autoload_with=kkbox_engine)
# full_test = Table('full_test', metadata, autoload=True,
#                   autoload_with=kkbox_engine)

# stmt1 = 'SELECT * FROM full_train WHERE rowid = 1'
# result1 = kkbox_conn.execute(stmt1).fetchone()
# result1 = [x for x in result1]
# len(result1[1:-1])
# np.asarray(result1, dtype=float)

# stmt2 = 'SELECT max(rowid) FROM full_train'
# result2 = kkbox_conn.execute(stmt2).scalar()
# result2

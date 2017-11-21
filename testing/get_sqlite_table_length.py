from sys import argv
from sqlalchemy import create_engine

_, db_name, *table_names = argv

sqlite_url = 'sqlite:///' + db_name
conn = create_engine(sqlite_url).connect()


def get_table_length(conn, table_name):
    length = conn.execute(
        'SELECT max(rowid) FROM' + table_name).scalar()
    return length


for name in table_names:
    length = get_table_length(conn, name)
    print("Table %s has %i rows" % (name, length))

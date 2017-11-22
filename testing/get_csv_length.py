from sys import argv

script, *file_path = argv


def get_dataset_length(file_path, had_header=True):
    """Retrieve file length of a large csv file"""
    with open(file_path, 'r') as f:
        length = 0
        for _ in f:
            length += 1
        length = length - had_header
    return length


for path in file_path:
    length = get_dataset_length(path)
    print('File "%s" has %i rows in the datafile.' % (path, length))

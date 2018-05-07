import os

if 'QNN_DB_PATH' in os.environ:
    DB_PATH = os.environ['QNN_DB_PATH']
else:
    DB_PATH = 'db'

if 'QNN_DATA_PATH' in os.environ:
    DATA_PATH = os.environ['QNN_DATA_PATH']
else:
    DATA_PATH = 'data'

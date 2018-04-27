import os

if 'DB_PATH' in os.environ:
    DB_PATH = os.environ['DB_PATH']
else:
    DB_PATH = 'db'

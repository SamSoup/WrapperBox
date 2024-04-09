import os


def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

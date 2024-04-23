import os
import json


def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filename: str):
    with open(filename, "r") as handle:
        data = json.load(handle)

    return data

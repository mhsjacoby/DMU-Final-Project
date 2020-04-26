"""
MyFunctions.py
Maggie Jacoby 2020

Basic functions I use for reading and writing data
"""

import os


def mylistdir(directory, bit='', end=True):
    filelist = os.listdir(directory)
    if end:
        return [x for x in filelist if x.endswith(f'{bit}') and not x.startswith('.') and not 'Icon' in x]
    else:
            return [x for x in filelist if x.startswith(f'{bit}') and not x.startswith('.') and not 'Icon' in x]

def make_storage_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir    
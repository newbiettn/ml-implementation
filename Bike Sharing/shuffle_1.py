################################################################################
# shuffle_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          07 May 2021
# DESCRIPTION:      Compress data to reduce its size to fit into memory and then shuffle it
#                   memory using zlib and random.shuffle().
################################################################################
import zlib
import random
import os


def ram_shuffle(filename_in, filename_out, header=True):
    with open(filename_in, 'rb') as f:
        zlines = [zlib.compress(r) for r in f]
        if header:
            first_row = zlines.pop(0)
    random.shuffle(zlines)
    with open(filename_out, 'wb') as f:
        if header:
            f.write(zlib.decompress(first_row))
        for zl in zlines:
            f.write(zlib.decompress(zl))


local_path = os.getcwd()
SOURCE = 'Bike Sharing/datasets/hour.csv'
OUT = 'Bike Sharing/datasets/hour_ram_shuffled.csv'
ram_shuffle(filename_in=local_path + '/' + SOURCE,
            filename_out=local_path + '/' + OUT,
            header=True)

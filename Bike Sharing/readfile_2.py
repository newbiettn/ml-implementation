################################################################################
# readfile_2.py
# AUTHOR:           NGOC TRAN
# CREATED:          07 May 2021
# DESCRIPTION:      Stream data from CSV file using pandas lib.
################################################################################
import pandas as pd
import os

local_path = os.getcwd()
SOURCE = 'Bike Sharing/datasets/hour.csv'
CHUNK_SIZE = 1000  # number of rows has to return at each iteration

with open(local_path + '/' + SOURCE) as stream:
    iterator = pd.read_csv(stream, chunksize=CHUNK_SIZE)
    for row in iterator:
        print ('Size of uploaded chunk: %i instance, %i features' %(row.shape))
    print ('Sample values:  \n%s' % str(row.iloc[0]))


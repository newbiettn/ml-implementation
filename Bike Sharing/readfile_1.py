################################################################################
# readfile_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          07 May 2021
# DESCRIPTION:      Stream data from CSV file using built-in utilities.
################################################################################
import csv
import os

local_path = os.getcwd()
SOURCE = 'Bike Sharing/datasets/hour.csv'
SEP = ','

with open(local_path + '/' + SOURCE) as R:
    csv_reader = csv.reader(R, delimiter=SEP)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            header = row
        else:
            line_count += 1
    print('Total rows: %i' % line_count)

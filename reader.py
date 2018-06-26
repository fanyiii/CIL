import numpy as np
from pathlib import Path
from random import *
import math


class Reader():

    def __init__(self):
        self.highest_col_index = 1000
        self.highest_row_index = 10000
        self.X_train = np.zeros(([self.highest_row_index, self.highest_col_index]))
        self.X_train_as_list = []
        self.X_validate_as_list = []
        self.X_validate = np.zeros(([self.highest_row_index, self.highest_col_index]))
        self.num_rows = 1176953
        self.file_name = "X_train.txt"
        self.train_file = Path(self.file_name)

    def convert_to_surprise_representation(self, path):
        X_train_as_list = ""
        rows_per_percent = self.num_rows / 100
        with open(path) as csvfile:
            i = 0
            percentage = 10            
            for line in csvfile:
                if (i == 0):
                    i += 1
                    continue
                i += 1
                tup = line.split(',')
                (row, col) = self.parse_index(tup[0])
                X_train_as_list += str(row) + " ; " + str(col) + " ; " + tup[1][:-1] + " ; \n"
                
                if (i >= percentage * rows_per_percent):
                    print(percentage, " % done")
                    percentage += 10
            csvfile.close()

        with open("surprise_format", "w") as newfile:
            newfile.write(X_train_as_list)
        newfile.close()
                

    def read_dataset(self, path, validation_percentage):
        self.X_train = np.zeros(([self.highest_row_index, self.highest_col_index]))
        self.X_validate = np.zeros(([self.highest_row_index, self.highest_col_index]))
        rows_per_percent = self.num_rows / 100

        """if self.train_file.exists():
            self.X = np.fromfile(self.file_name)
            return"""

        with open(path) as csvfile:
            i = 0
            percentage = 10
            num_eval = 0.
            num_train = 0.
            for line in csvfile:
                if (i == 0):
                    i += 1
                    continue
                i += 1
                tup = line.split(',')
                (row, col) = self.parse_index(tup[0])
                """if (row > self.highest_row_index):
                    print("resize performed: row")
                    self.highest_row_index = row
                    X.resize([self.highest_row_index, self.highest_col_index])
                if (col > self.highest_col_index):
                    print("resize performed: col")
                    self.highest_col_index = col
                    X.resize([self.highest_row_index, self.highest_col_index])"""
                if (random() < validation_percentage):
                    self.X_validate[row, col] = int(tup[1])
                    num_eval += 1
                else:
                    self.X_train[row, col] = int(tup[1])
                    num_train += 1

                if (i >= percentage * rows_per_percent):
                    print(percentage, " % done")
                    percentage += 10

            csvfile.close()
        print("Set counts: Validation: ", num_eval, ", Train: ", num_train)
        print("Real Validation percentage: ", num_eval/self.num_rows)
        # self.X.tofile(self.file_name)

    def parse_index(self, index):
        index = index[1:]
        ind = index.split('c')
        row = int(ind[0][:-1]) - 1
        col = int(ind[1]) - 1
        if (math.isnan(row) or math.isnan(col)):
        	print(index)
        	print(row)
        	print(col)
        	raise ValueError("What the fuck is going on")
        return (row, col)

    """def read_to_be_predicted(self, path):
        self.indices_to_predict = {}
        with open(path) as csvfile:
            firstline = True
            for line in csvfile:
                if (firstline):
                    firstline = False
                    continue
                tup = line.split(',')
                self.indices_to_predict.update({tup[0] : self.parse_index(tup[0])})
            csvfile.close()"""


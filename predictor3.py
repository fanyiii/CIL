import numpy as np
import csv
from reader import Reader
import operator
from pathlib import Path
import math
import random



class NeighborInitialization():
    def __init__(self, trainset_path, validation_percentage):
        self.data_path = trainset_path
        self.validation_percentage = validation_percentage
        self.reader = Reader()

    def compute_error(self,user_a, user_b):
        """DEPRECATED, only for descriptive purpose: Describes how similar two users rate items, we want to look at users who tend to rate similar"""
        error = 0.
        ratings_of_user_a = self.X_train[user_a,:]
        ratings_of_user_b = self.X_train[user_b,:]
        for i in range(len(ratings_of_user_a)):
            if (ratings_of_user_a[i] < 1 or ratings_of_user_b[i] < 1):
                error += 4  #if we don't know both ratings, then we assume the distance to be maximal
            else:
                error += math.sqrt((ratings_of_user_a[i] - ratings_of_user_b[i])**2)
        return error

    def compute_error_2(self,user_a, user_b):
        """Does the same as compute_error, but is much faster"""        
        if (user_a == user_b):
            return 0
        
        ratings_of_user_a = self.X_train[user_a,:]
        ratings_of_user_b = self.X_train[user_b,:]
        res_list = np.abs(ratings_of_user_a - ratings_of_user_b) * self.X_train_mask[user_a,:] * self.X_train_mask[user_b,:]
        return np.sum(res_list) + 4*(len(ratings_of_user_a) - np.count_nonzero(ratings_of_user_a*ratings_of_user_b))
        
                
    def predict_concerning_neighbors(self, user, n):
        """
        - For a given user, look for users which tend to rate similar and based on their rating of a certain item propose a rating for this item for the current user
        """
        X_error = np.zeros(np.shape(self.X_train_mask))
        for i in range(len(self.X_train[:, 0])):
            #X_error[i,j] is set to the difference between user "user" and user "i" for all j
            X_error[i,:] = self.X_train_mask[i,:] * self.compute_error_2(i, user)
        #for unknown ratings X_error is assumed to be maximal (if all items are known, the maximal possible error is 4000 in our case)
        X_error[X_error == 0] = 4096
        num_relevant_users = 10
        
        #sorted_errors[i,j] contains the index of the user with i-th best similarity to the current user. Note that if error < 4000, then the user with that index HAS a rating for item j.
        sorted_errors = np.argsort(X_error, axis=0)[:num_relevant_users,:]

        avg = np.zeros((n,))
        for i in range(num_relevant_users):
            for j in range(n):
                temp = self.X_train[sorted_errors[i][j]][j]
                avg[j] += temp if temp > 0 else 3  #if not enough "useful" ratings for item j exist, then we bend towards the average
        avg /= num_relevant_users
        res = np.rint(avg) #round to nearest integer
        return res
        

    def do_work(self):
        self.reader.read_dataset(self.data_path, validation_percentage=self.validation_percentage)
        self.X_train = np.matrix.copy(self.reader.X_train)  #Contains ratings where ratings are present and 0 everywhere else
        d, n = np.shape(self.X_train)
        self.X_train_mask = np.zeros((d,n))
        for d_ in range(d):
            for n_ in range(n):
                if (self.X_train[d_][n_] > 0):
                    self.X_train_mask[d_][n_] = 1

        
        #unique, counts = np.unique(self.X_train, return_counts=True)
        #print(dict(zip(unique,counts)))        #Note: Running these shows how many items are given a specific rating. Note that good ratings are more probable than bad ratings.
        
        for d_ in range(d):
            if (d_ < 10 or d_ % 10 == 0):
                print(d_)
            self.X_train[d_][:] -= (self.X_train_mask[d_][:] - np.ones((n,))) * self.predict_concerning_neighbors(d_,n)     #Only want to chang self.X_train where no rating (==0) is present
        
        
        with open("ExtendedTrainSet.csv", "w") as newfile:
            writer = csv.writer(newfile)
            writer.writerow(["Id", "Prediction"])
            for d_ in range(d):
                for n_ in range(n):
                    writer.writerow(["r" + str(d_+1) + "_c" + str(n_+1), self.X_train[d_][n_]])

            newfile.close()

     
        
        

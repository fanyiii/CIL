'''
Generate predicted ratings by averaging the output of three models:
user-user k-NN model based on the Pearson correlation similarity,
item-item k-NN model based on the Pearson correlation similarity,
and a SVD++ model.

The fusion schema is chosen based on results of experiments.
'''

import pandas as pd
import numpy as np
import os

from surprise import Reader
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import Dataset

from sklearn.utils import shuffle

from reader_writer import ReaderWriter

# Read training and testing data
reader_writer = ReaderWriter()
if not os.path.isfile('separated_ratings.csv'):
    reader_writer.change_file_format('data_train.csv', 'separated_ratings.csv')
else: 
    print("separated_ratings already exists")

if not os.path.isfile('separated_sample_submission.csv'):
    reader_writer.change_file_format('sampleSubmission.csv', 'separated_sample_submission.csv')
else: 
    print("separated_sample_submission already exists")

train = shuffle(pd.read_csv('separated_ratings.csv'))
reader = Reader(rating_scale=(1, 5))
train = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
train = train.build_full_trainset()

test = pd.read_csv('separated_sample_submission.csv')

# Train a user-user k-NN model based on the Pearson correlation similarity
print("Train a user-user k-NN model based on the Pearson correlation similarity")
sim_options = {'name': 'pearson_baseline'}
knn_user_pearson = KNNBaseline(sim_options=sim_options)
knn_user_pearson.fit(train)

# Use the model to predict ratings
def knn_user_pearson_est(row):
    return knn_user_pearson.predict(row.userId, row.movieId).est
print("Use user-user k-NN model to predict ratings")
test["knn_user_pearson"] = test.apply(knn_user_pearson_est, axis=1)
del knn_user_pearson

# Train a item-item k-NN model based on the Pearson correlation similarity

print("Train a item-item k-NN model based on the Pearson correlation similarity")
sim_options = {'name': 'pearson_baseline', 'user_based': False}
knn_item_pearson = KNNBaseline(sim_options=sim_options)
knn_item_pearson.fit(train)

# Use the model to predict ratings
def knn_item_pearson_est(row):
    return knn_item_pearson.predict(row.userId, row.movieId).est
print("Use item-item k-NN model to predict ratings")
test["knn_item_pearson"] = test.apply(knn_item_pearson_est, axis=1)
del knn_item_pearson

# Train a SVD++ model

print("Train a SVD++ model")
svd = SVDpp(n_epochs=10)
svd.fit(train)

# Use the model to predict ratings
def svd_est(row):
    return svd.predict(row.userId, row.movieId).est
print("Use SVD++ model to predict ratings")
test["svd"] = test.apply(svd_est, axis=1)

# Calculate average ratings of the those three outputs
predicted_ratings = (test.knn_user_pearson + test.knn_item_pearson + test.svd) / 3

# Write predicted ratings to file
reader_writer.write_to_file(predicted_ratings,'sampleSubmission.csv','predicted_ratings.csv')

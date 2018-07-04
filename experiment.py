import pandas as pd

from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import Dataset
from surprise import accuracy
from surprise import dump

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import os

# Generate a train data set and a validation data set
if os.path.isfile('train.csv') and os.path.isfile('valid.csv'):
    print("train and valid data already exists")
    train = pd.read_csv('train.csv')
    valid = pd.read_csv('valid.csv')
else:
    if not os.path.isfile('separated_ratings.csv'):
        reader_writer.change_file_format('data_train.csv', 'separated_ratings.csv') 
    ratings = pd.read_csv('separated_ratings.csv')
    ratings = shuffle(ratings)
    msk = np.random.rand(len(ratings)) < 0.8
    train = ratings[msk]
    valid = ratings[~msk]

    train.to_csv('train.csv', index=False)
    valid.to_csv('valid.csv', index=False)  

# Train KNN mo#dels and SVD mo#dels with different settings
reader = Reader(rating_scale=(1, 5))
train = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
train = train.build_full_trainset()

knn = KNNBaseline()
knn.fit(train)

sim_options = {'user_based': False}
knn_item = KNNBaseline(sim_options=sim_options)
knn_item.fit(train)

sim_options = {'name': 'pearson_baseline'}
knn_pearson = KNNBaseline(sim_options=sim_options)
knn_pearson.fit(train)

sim_options = {'name': 'pearson_baseline', 'user_based': False}
knn_item_pearson = KNNBaseline(sim_options=sim_options)
knn_item_pearson.fit(train)

svd = SVDpp(n_epochs=10)
svd.fit(train)

svd_10 = SVDpp(n_factors=10, n_epochs=10)
svd_10.fit(train)

svd_30 = SVDpp(n_factors=30, n_epochs=10)
svd_30.fit(train)

# Test mse of those basic mo#dels
print("User-user KNN mo#del based on Mean Squared Difference similarity")
def knn_est(row):
    return knn.predict(row.userId, row.movieId).est
valid["knn"] = valid.apply(knn_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.knn)))
#del knn

print("Item-item KNN mo#del based on Mean Squared Difference similarity")
def knn_item_est(row):
    return knn_item.predict(row.userId, row.movieId).est
valid["knn_item"] = valid.apply(knn_item_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.knn_item)))
#del knn_item

print("User-user KNN mo#del based on Pearson correlation similarity")
def knn_pearson_est(row):
    return knn_pearson.predict(row.userId, row.movieId).est
valid["knn_pearson"] = valid.apply(knn_pearson_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.knn_pearson)))
#del knn_pearson

print("Item-item KNN mo#del based on Pearson correlation similarity")
def knn_item_pearson_est(row):
    return knn_item_pearson.predict(row.userId, row.movieId).est
valid["knn_item_pearson"] = valid.apply(knn_item_pearson_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.knn_item_pearson)))
#del knn_item_pearson

print("SVD++ mo#del factors=20")
def svd_est(row):
    return svd.predict(row.userId, row.movieId).est
valid["svd"] = valid.apply(svd_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.svd)))
#del svd

print("SVD++ mo#del factors=10")
def svd_10_est(row):
    return svd_10.predict(row.userId, row.movieId).est
valid["svd_10"] = valid.apply(svd_10_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.svd_10)))
#del svd_10

print("SVD++ mo#del factors=30")
def svd_30_est(row):
    return svd_30.predict(row.userId, row.movieId).est
valid["svd_30"] = valid.apply(svd_30_est, axis=1)
print(sqrt(mean_squared_error(valid.rating, valid.svd_30)))
#del svd_30

# Test different ways to fuse those basic mo#dels
print("knn user-user msd + knn item-item msd")
print(sqrt(mean_squared_error(valid.rating, (valid.knn + valid.knn_item) / 2)))

print("knn user-user pearson + knn item-item pearson")
print(sqrt(mean_squared_error(valid.rating, (valid.knn_pearson + valid.knn_item_pearson) / 2)))

print("knn user-user msd + knn user-user pearson")
print(sqrt(mean_squared_error(valid.rating, (valid.knn + valid.knn_pearson) / 2)))

print("knn item-item msd + knn item-item pearson")
print(sqrt(mean_squared_error(valid.rating, (valid.knn_item + valid.knn_item_pearson) / 2)))

print("knn user-user msd + knn item-item msd + knn user-user pearson + knn item-item pearson")
valid["knn_fusion"] = (valid.knn + valid.knn_pearson + valid.knn_item + valid.knn_item_pearson) / 4
print(sqrt(mean_squared_error(valid.rating, valid.knn_fusion)))

print("svd factors=10 + svd factors=20")
print(sqrt(mean_squared_error(valid.rating, (valid.svd_10 + valid.svd) / 2)))

print("svd factors=20 + svd factors=30")
print(sqrt(mean_squared_error(valid.rating, (valid.svd_30 + valid.svd) / 2)))

print("svd factors=10 + svd factors=30")
print(sqrt(mean_squared_error(valid.rating, (valid.svd_10 + valid.svd_30) / 2)))

print("svd factors=10 + svd factors=20 + svd factors=30")
valid["svd_fusion"] = (valid.svd_10 + valid.svd + valid.svd_30) / 3
print(sqrt(mean_squared_error(valid.rating, valid.svd_fusion)))

print("svd factors=20 + knn user-user pearson + knn item-item pearson")
valid['fusion'] = (valid.svd + valid.knn_pearson + valid.knn_item_pearson) / 3
print(sqrt(mean_squared_error(valid.rating, valid.fusion)))

print("svd factors=20 + knn item-item msd + knn item-item pearson")
print(sqrt(mean_squared_error(valid.rating, (valid.svd + valid.knn_item + valid.knn_item_pearson) / 3)))

print("fusion of all models")
valid["fusion_all"] = (valid.knn_fusion * 4 + valid.svd_fusion * 3) / 7
print(sqrt(mean_squared_error(valid.rating, valid.fusion_all)))

# Test the effect of rounding
def custom_round(n):
    if n % 1 <= 0.01 or n % 1 >= 0.99:
        return int(round(n))
    else:
        return n

print("round knn item-item pearson")
print(sqrt(mean_squared_error(valid.rating, valid.knn_item_pearson.apply(lambda n: custom_round(n)))))

print("round knn fusion")
print(sqrt(mean_squared_error(valid.rating, valid.knn_fusion.apply(lambda n: custom_round(n)))))  

print("round svd fusion")
print(sqrt(mean_squared_error(valid.rating, valid.svd_fusion.apply(lambda n: custom_round(n)))))

print("round svd factors=20 + knn user-user pearson + knn item-item pearson")
print(sqrt(mean_squared_error(valid.rating, valid.fusion.apply(lambda n: custom_round(n)))))

print("round fusion all")
print(sqrt(mean_squared_error(valid.rating, valid.fusion_all.apply(lambda n: custom_round(n)))))

'''
Result of the experiment:
This two sets of configuration perform well:
svd factors=20 + knn user-user pearson + knn item-item pearson
0.9875974059700611
svd factors=20 + knn item-item msd + knn item-item pearson
0.9881138333034675

Rounding does not reduce mse.

Generated output for reference:
train and valid data already exists
User-user KNN mo#del based on Mean Squared Difference similarity
1.0109252052352078
Item-item KNN mo#del based on Mean Squared Difference similarity
0.9958455935063554
User-user KNN mo#del based on Pearson correlation similarity
0.9999969138465615
Item-item KNN mo#del based on Pearson correlation similarity
0.9912122336252883
SVD++ mo#del factors=20
1.0031305093088387
SVD++ mo#del factors=10
1.0037882960120001
SVD++ mo#del factors=30
1.0039225718184417
knn user-user msd + knn item-item msd
0.9969156644265307
knn user-user pearson + knn item-item pearson
0.9901539770400012
knn user-user msd + knn user-user pearson
0.9975048878879126
knn item-item msd + knn item-item pearson
0.9886929476311236
knn user-user msd + knn item-item msd + knn user-user pearson + knn item-item pearson
0.9900447504114451
svd factors=10 + svd factors=20
1.0010876062501912
svd factors=20 + svd factors=30
1.0001955383407333
svd factors=10 + svd factors=30
1.0009351392585901
svd factors=10 + svd factors=20 + svd factors=30
0.999779551318123
svd factors=20 + knn user-user pearson + knn item-item pearson
0.9875974059700611
svd factors=20 + knn item-item msd + knn item-item pearson
0.9881138333034675
fusion of all models
0.9899546938788643
round knn item-item pearson
0.9912129442364213
round knn fusion
0.9900419713598395
round svd fusion
0.9997783775670502
round svd factors=20 + knn user-user pearson + knn item-item pearson
0.9875981013697378
round fusion all
0.989954716228364
'''
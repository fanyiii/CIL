import pandas as pd
import numpy as np 
import csv
import os.path

from reader_nn import Reader
reader = Reader()

from neural_net import NeuralNet
neural_net = NeuralNet()

if not os.path.isfile('separated_ratings.csv'):
	reader.change_file_format()
else: 
	print("separated_ratings already exists")

separated_ratings = pd.read_csv('separated_ratings.csv')

users = separated_ratings.userId.unique()
movies = separated_ratings.movieId.unique()
rating = separated_ratings.rating.unique()
print("Users: ", len(users), ", Movies: ", len(movies), ", Ratings: ", len(rating))

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}
movieidx2id = {i:o for i,o in enumerate(movies)}

new_ratings = separated_ratings.copy()
new_ratings.movieId = separated_ratings.movieId.apply(lambda x: movieid2idx[x])
new_ratings.userId = separated_ratings.userId.apply(lambda x: userid2idx[x])

## split data into training and validation set, 80% in training, 20% in validation
msk = np.random.rand(len(new_ratings)) < 0.8
train = new_ratings[msk]
valid = new_ratings[~msk]

print("training data: ", train.shape, ",validation data: ", valid.shape)

#neural_net.baseline(train, valid, users, movies)
#neural_net.multilayer_LSTM(train, valid, users, movies)
neural_net.bidirectional_LSTM(train, valid, users, movies)
#neural_net.bidirectional_LSTM_2(train, valid, users, movies)
























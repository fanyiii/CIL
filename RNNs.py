import pandas as pd
import numpy as np 
import csv
import os.path

from reader_writer import ReaderWriter
reader_writer = ReaderWriter()

from neural_net import NeuralNet
neural_net = NeuralNet()

if not os.path.isfile('separated_ratings.csv'):
	reader_writer.change_file_format('data_train.csv', 'separated_ratings.csv')
else: 
	print("separated_ratings already exists")

if not os.path.isfile('separated_sample_submission.csv'):
	reader_writer.change_file_format('sample_submission.csv', 'separated_sample_submission.csv')
else: 
	print("separated_sample_submission already exists")

separated_ratings = pd.read_csv('separated_ratings.csv')
separated_sample_submission = pd.read_csv('separated_sample_submission.csv')
separated_sample_submission.rating = 0

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
predicted_ratings = neural_net.bidirectional_LSTM(train, valid, users, movies, separated_sample_submission)
#neural_net.bidirectional_LSTM_2(train, valid, users, movies)

reader_writer.write_to_file(predicted_ratings,'sample_submission.csv','predicted_ratings.csv')
























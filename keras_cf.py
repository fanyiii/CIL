# Reference: https://nipunbatra.github.io/blog/2017/recommend-keras.html

import pandas as pd
import numpy as np 
import csv
import os.path

from sklearn.utils import shuffle
import keras
from keras.layers import Embedding, Reshape, Dropout, Dense, Merge
from keras.models import Sequential

K_FACTORS = 50

from reader_writer import ReaderWriter
reader_writer = ReaderWriter()

if not os.path.isfile('separated_ratings.csv'):
    reader_writer.change_file_format('data_train.csv', 'separated_ratings.csv')
else: 
    print("separated_ratings already exists")

if not os.path.isfile('separated_sample_submission.csv'):
    reader_writer.change_file_format('sampleSubmission.csv', 'separated_sample_submission.csv')
else: 
    print("separated_sample_submission already exists")

separated_ratings = pd.read_csv('separated_ratings.csv')
separated_sample_submission = pd.read_csv('separated_sample_submission.csv')
separated_sample_submission.rating = 0

to_predict = separated_sample_submission

users = separated_ratings.userId.unique()
movies = separated_ratings.movieId.unique()
rating = separated_ratings.rating.unique()
print("Users: ", len(users), ", Movies: ", len(movies), ", Ratings: ", len(rating))

ratings = shuffle(separated_ratings)
# For output final result for submission:
train = ratings

# Define model
def cf_model(n_users, m_items, k_factors):
    dropout_rate = 0.2

    P = Sequential()
    P.add(Embedding(n_users, k_factors, input_length=1))
    P.add(Reshape((k_factors,)))
    P.add(Dropout(dropout_rate))

    Q = Sequential()
    Q.add(Embedding(m_items, k_factors, input_length=1))
    Q.add(Reshape((k_factors,)))
    Q.add(Dropout(dropout_rate))

    model = Sequential()
    model.add(Merge([P, Q], mode='concat',name='Concat'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mse', optimizer='adam')
    return model

model = cf_model(len(users), len(movies), K_FACTORS)
model.fit([train.userId, train.movieId], train.rating, batch_size=64, epochs=10, verbose=1)

'''
# For validation:
msk = np.random.rand(len(ratings)) < 0.8
train = ratings[msk]
valid = ratings[~msk]
model_validate = cf_model(len(users), len(movies), K_FACTORS)
model_validate.fit([train.userId, train.movieId], train.rating, batch_size=64, epochs=20, verbose=1, validation_data=([valid.userId, valid.movieId], valid.rating))
'''

predicted_ratings = model.predict([to_predict.userId, to_predict.movieId])

reader_writer.write_to_file(predicted_ratings,'sampleSubmission.csv','nn_relu.csv')





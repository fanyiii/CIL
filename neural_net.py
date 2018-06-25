import pandas as pd
import numpy as np 

from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy import stats
from scipy.spatial.distance import pdist

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from itertools import cycle

import string

from collections import Counter

from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.layers.core import Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras.regularizers import l1, l2

from keras.layers.recurrent import GRU, LSTM

from keras import backend as K

class NeuralNet():

	##Simple neural net
	def baseline(self, train, valid, users, movies):
		
		##Bias layer
		user_input = Input(shape=(1,), dtype='int64', name='user_input')
		movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
		user_embed = Embedding(len(users), 50, input_length =1)(user_input)
		movie_embed = Embedding(len(movies), 50, input_length =1)(movie_input)

		x = concatenate([user_embed, movie_embed], axis=-1)

		x = Flatten()(x)
		x = Dropout(0.3)(x)
		x = Dense(70, activation='relu')(x)
		x = Dropout(0.75)(x)
		x = Dense(1)(x)
		nn = Model([user_input, movie_input], x)
		nn.compile(Adam(0.001), loss='mse')

		BASELINE = nn.fit(
			[train.userId, train.movieId], 
			train.rating, 
			batch_size=64, 
			nb_epoch=20, 
		    validation_data=([valid.userId, valid.movieId], valid.rating))

		plt.plot(BASELINE.history['loss'], label = 'loss')
		plt.plot(BASELINE.history['val_loss'], label = 'val_loss')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('MSE loss')
		plt.show()

	def multilayer_LSTM(self, train, valid, users, movies):
		# Multi layer LSTM model

		#Bias layer
		user_input = Input(shape=(1,), dtype='int64', name='user_input')
		movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
		user_embed = Embedding(len(users), 50, input_length =1)(user_input)
		movie_embed = Embedding(len(movies), 50, input_length =1)(movie_input)
		x = concatenate([user_embed, movie_embed], axis=-1)

		x = Dropout(0.75)(x)
		BatchNormalization()
		x = LSTM(50)(x)
		x = Dropout(0.75)(x)
		BatchNormalization()
		x = Dense(1)(x)
		LSTM_nn = Model([user_input, movie_input], x)
		LSTM_nn.compile(Adam(0.001), loss='mse')

		LSTM_mult_history = LSTM_nn.fit(
			[train.userId, train.movieId], 
			train.rating, 
			batch_size=64, 
			nb_epoch=15, 
			validation_data=([valid.userId, valid.movieId], valid.rating))

		plt.clf()
		plt.figure(figsize = (10,7))
		plt.plot(LSTM_mult_history.history['loss'], label = 'LSTMloss')
		plt.plot(LSTM_mult_history.history['val_loss'], label = 'LSTMval_loss')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('MSE loss')
		plt.show()

	def bidirectional_LSTM(self, train, valid, users, movies):
		# Bidirectional LSTM model with regularization
		embedding_output = 100
		dropout = 0.75
		nodes = 50
		batch_size = 50
		epochs = 10

		#Bias layer
		user_input = Input(shape=(1,), dtype='int64', name='user_input')
		movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
		user_embed = Embedding(len(users), embedding_output, input_length =1)(user_input)
		movie_embed = Embedding(len(movies), embedding_output, input_length =1)(movie_input)
		x = concatenate([user_embed, movie_embed], axis=-1)

		

		print("Dropout: ", dropout, ", Nodes: ", nodes, ", Batch_size: ", batch_size, ", Epochs: ", epochs)

		x = Dropout(dropout)(x)
		BatchNormalization()
		x_fwd = LSTM(nodes)(x) # nodes
		x_bwd = LSTM(nodes, go_backwards = True)(x) # nodes backwards
		x_bdir = concatenate([x_fwd, x_bwd], axis=-1)
		x = Dropout(dropout)(x_bdir)
		BatchNormalization()
		x = Dense(1)(x)
		LSTM50_bdir = Model([user_input, movie_input], x)
		LSTM50_bdir.compile(Adam(0.001), loss='mse')
	
		LSTM_bi_history = LSTM50_bdir.fit(
			[train.userId, train.movieId], 
			train.rating, 
			batch_size=batch_size,
			nb_epoch=epochs, 
   			validation_data=([valid.userId, valid.movieId], valid.rating))

		plt.clf()
		plt.figure(figsize = (10,7))
		plt.plot(LSTM_bi_history.history['loss'], label = 'BiLSTMloss')
		plt.plot(LSTM_bi_history.history['val_loss'], label = 'BiLSTMval_loss')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('MSE loss')
		plt.show()

	def bidirectional_LSTM_2(self, train, valid, users, movies):
		user_input = Input(shape=(1,), dtype='int64', name='user_input')
		movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
		user_embed = Embedding(len(users), 50, input_length =1)(user_input)
		movie_embed = Embedding(len(movies), 50, input_length =1)(movie_input)
		x = concatenate([user_embed, movie_embed], axis=-1)

		x = Dropout(0.75)(x)
		BatchNormalization()
		x_fwd = LSTM(40)(x)
		x_bwd = LSTM(40, go_backwards = True)(x)
		x_bdir = concatenate([x_fwd, x_bwd], axis=-1)
		x = Dropout(0.75)(x_bdir)
		BatchNormalization()
		x = Dense(1)(x)
		LSTM40_bdir = Model([user_input, movie_input], x)
		LSTM40_bdir.compile(Adam(0.001), loss='mse')

		user_input = Input(shape=(1,), dtype='int64', name='user_input')
		movie_input = Input(shape = (1,), dtype = 'int64', name = 'movie_input')
		user_embed = Embedding(len(users), 50, input_length =1)(user_input)
		movie_embed = Embedding(len(movies), 50, input_length =1)(movie_input)
		x = concatenate([user_embed, movie_embed], axis=-1)

		x = Dropout(0.75)(x)
		BatchNormalization()
		x_fwd = LSTM(40)(x)
		x_bwd = LSTM(40, go_backwards = True)(x)
		x_bdir = concatenate([x_fwd, x_bwd], axis=-1)
		Activation_model = Model([user_input, movie_input], x_bdir)

		for layer in zip(LSTM40_bdir.layers[:-2], Activation_model.layers):
		    # the new weights are the second element in the tuple
		    layer[1].set_weights([x for x in layer[0].get_weights()])

		Activation_model.compile(Adam(0.001), loss='mse')
		activations = Activation_model.predict([valid.userId, valid.movieId])
		valid.to_csv('predicted_ratings.csv')
		#np.save('activations.npy', activations)
		#X_50_iter5000 = np.load('data/MovieLens/X_50_iter5000.npy')














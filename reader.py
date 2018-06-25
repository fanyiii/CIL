import pandas as pd
import numpy as np 
import csv

class Reader():

	def change_file_format(self):
		org_ratings = pd.read_csv('data_train.csv')
		org_ratings.head()

		users_movies = org_ratings.Id
		users = np.zeros(users_movies.shape)
		user_count = 0
		movies = np.zeros(users_movies.shape)
		movie_count = 0

		ratings = org_ratings.Prediction

		for item in users_movies:
			tup = item.split('c')
			user = int(tup[0][1:-1])
			users[user_count] = user
			user_count = user_count+1
			movie = int(tup[1])
			movies[movie_count] = movie
			movie_count = movie_count+1

		with open('separated_ratings.csv', "w") as newfile:
		    writer = csv.writer(newfile)
		    writer.writerow(["userId", "movieId", "rating"])
		    for i in range(users_movies.shape[0]):
		    	writer.writerow([users[i],movies[i], ratings[i]])
		    	if i % 100000 == 0:
		    		print("finished writing ", i, "of ", users_movies.shape[0], " lines")


		print("finished writing")
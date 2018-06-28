import pandas as pd
import numpy as np 
import csv

class ReaderWriter():

	def change_file_format(self, file_name, file_name_2):
		org_ratings = pd.read_csv(file_name)
		org_ratings.head()

		users_movies = org_ratings.Id
		users = np.zeros(users_movies.shape)
		user_count = 0
		movies = np.zeros(users_movies.shape)
		movie_count = 0

		ratings = org_ratings.Prediction

		for item in users_movies:
			tup = item.split('c')
			user = int(tup[0][1:-1])-1
			users[user_count] = user
			user_count = user_count+1
			movie = int(tup[1])-1
			movies[movie_count] = movie
			movie_count = movie_count+1

		with open(file_name_2, "w") as newfile:
		    writer = csv.writer(newfile)
		    writer.writerow(["userId", "movieId", "rating"])
		    for i in range(users_movies.shape[0]):
		    	writer.writerow([users[i],movies[i], ratings[i]])
		    	if i % 100000 == 0:
		    		print("finished writing ", i, "of ", users_movies.shape[0], " lines")


		print("finished writing separated file")

	def write_to_file(self, data, index_file, file_name):
		indexes = pd.read_csv(index_file).Id
		with open(file_name, "w") as newfile:
			writer = csv.writer(newfile)
			writer.writerow(["Id", "Prediction"])
			for i in range(data.shape[0]):
				writer.writerow([indexes[i], int(data[i][0])])
				if i % 10000 == 0:
					print("finished writing ", i, "of ", data.shape[0], " lines")

		print("finished writing prediction file")

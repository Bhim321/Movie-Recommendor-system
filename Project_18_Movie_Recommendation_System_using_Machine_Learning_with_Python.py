#!/usr/bin/env python
# coding: utf-8

# Importing the dependencies

# Bhim Yadav(203040021)

# In[ ]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Data Collection and Pre-Processing

# In[ ]:


# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('/content/movies.csv')


# In[ ]:


# printing the first 5 rows of the dataframe
movies_data.head()


# In[ ]:


# number of rows and columns in the data frame

movies_data.shape


# In[ ]:


# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[ ]:


# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[ ]:


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[ ]:


print(combined_features)


# In[ ]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[ ]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[ ]:


print(feature_vectors)


# Cosine Similarity

# In[ ]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[ ]:


print(similarity)


# In[ ]:


print(similarity.shape)


# Getting the movie name from the user

# In[ ]:


# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[ ]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[ ]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[ ]:


close_match = find_close_match[0]
print(close_match)


# In[ ]:


# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[ ]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[ ]:


len(similarity_score)


# In[ ]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[ ]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# Movie Recommendation Sytem

# In[ ]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:





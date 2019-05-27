# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:56:59 2019

@author: gaurav.pandey1
"""

#import numpy as np;
#import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = "https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv"
df = pd.read_csv(url)

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]

def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]
    
features = ['keywords','cast','genres','director']

for feature in features:
    df[feature] = df[feature].fillna('')
    
def combine_features(row):
    return row['keywords']+' '+row['cast']+' '+row['genres']+' '+row['director']

df["combine_features"] = df.apply(combine_features,axis=1)

#using CountVectorizer
#cv = CountVectorizer()
#count_matrix=cv.fit_transform(df["combine_features"])

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
count_matrix=tfidf.fit_transform(df["combine_features"])

cosin_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosin_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if(i>5):
        break
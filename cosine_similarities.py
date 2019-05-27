# -*- coding: utf-8 -*-
"""
Created on Sat May 18 08:56:14 2019

@author: gaurav.pandey1
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London","Paris Paris London"]

cv = CountVectorizer()
x=cv.fit_transform(text)
print(x.toarray())

similarity_score = cosine_similarity(x)

print(similarity_score)


#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd

file_path = './ratings.dat'

columns = ['userID', 'movieID', 'rating', 'timestamp']
ratings_data = pd.read_csv(file_path, sep = '::', names = columns, engine = 'python')

print(ratings_data.head())


# In[18]:


# 6040 x 3952 크기의 matrix 만듦.
# null 값 0으로 채우기까지!

ratings_matrix = np.zeros((6040, 3952))

def update_ratings_matrix(row):
    ratings_matrix[row['userID'] - 1, row['movieID'] - 1] = row['rating']

ratings_data.apply(update_ratings_matrix, axis=1)
print(ratings_matrix)


# In[103]:


# 3개 클러스터링 하기
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, n_init = 10)
kmeans.fit(ratings_matrix)

labels = kmeans.labels_

group, counts = np.unique(labels, return_counts = True)
cluster_counts = dict(zip(group, counts))

print(cluster_counts)


# In[104]:


# 데이터 이름붙이고 정리

group_people = {0: [], 1: [], 2: []}

for user_id, num in enumerate(labels):
    group_people[num].append(user_id)


# In[105]:


def average_rating(matrix, group_people):
    mean_ratings = matrix[group_people].mean(axis = 0)
    return mean_ratings


# In[106]:


def additive_utilitarian(matrix, group_people):
    sum_ratings = matrix[group_people].sum(axis = 0)
    return sum_ratings


# In[107]:


def simple_count(matrix, group_people):
    count_ratings = (matrix[group_people] > 0).sum(axis = 0)
    return count_ratings


# In[108]:


def approval_voting(matrix, group_people):
    approval_ratings = (matrix[group_people] >= 4).sum(axis = 0)
    return approval_ratings


# In[109]:


def borda_count(matrix, group_people):
    rankings = np.argsort(-matrix[group_people], axis = 1)
    borda_scores = np.zeros(matrix.shape[1])

    for r in rankings:
        for rank, movie_id in enumerate(r):
            borda_scores[movie_id] += (matrix.shape[1] - rank)

    return borda_scores


# In[119]:


def copeland_rule(matrix, group_people):
    group_people_matrix = matrix[group_people]
    num_items = group_people_matrix.shape[1]
    
    wins = np.zeros((num_items, num_items))
    for a in range(num_items):
        for b in range(a+1, num_items):
            a_ = np.sum(group_people_matrix[:, a] > group_people_matrix[:, b])
            b_ = np.sum(group_people_matrix[:, a] < group_people_matrix[:, b])
            if a_ > b_:
                wins[a, b] = 1
                wins[b, a] = -1
            elif a_ < b_:
                wins[a, b] = -1
                wins[b, a] = 1

    copeland_scores = np.sum(wins, axis=1)
    return copeland_scores


# In[120]:


algorithms = {
    'Average': average_rating,
    'Additive Utilitarian': additive_utilitarian,
    'Simple Count': simple_count,
    'Approval Voting': approval_voting,
    'Borda Count': borda_count,
    'Copeland Rule': copeland_rule
}


# In[ ]:


recommend = {}

for i in range(3):
    algo = {}
    
    for name in algorithms:
        algo[name] = []
    
    recommend[i] = algo

for num, users in group_people.items():
    for name, function in algorithms.items():
        scores = function(ratings_matrix, users)
        top_10 = np.argsort(-scores)[:10]
        recommend[num][name] = top_10

for num, recs in recommend.items():
    print(f'Group {num + 1}')
    for name, items in recs.items():
        print(f'{name}: {items}')


# In[ ]:





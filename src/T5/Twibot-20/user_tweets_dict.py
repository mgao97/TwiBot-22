import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

path1 = '/dev/shm/twi20/data/node.json'
with open(path1, 'r') as f:
    nodes = json.loads(f.read())
    nodes = pd.DataFrame(nodes)
    users = nodes[nodes.id.str.contains('^u')]
    tweets = nodes[nodes.id.str.contains('^t')]
    
users_index_to_uid = list(users['id'])
tweets_index_to_tid = list(tweets['id'])
uid_to_users_index = {x: i for i, x in enumerate(users_index_to_uid)}
tid_to_tweets_index = {x: i for i, x in enumerate(tweets_index_to_tid)}

path2 = '/dev/shm/twi20/data/edge.csv'
edge_data = pd.read_csv(path2)
edge_data = edge_data[edge_data['relation'] == 'post']

edge_data['source_id'] = list(map(lambda x: uid_to_users_index[x], edge_data['source_id'].values))
edge_data['target_id'] = list(map(lambda x: tid_to_tweets_index[x], edge_data['target_id'].values))
edge_data = edge_data.reset_index(drop=True)
user_to_posts = {i: [] for i in range(len(users))}

for i in tqdm(range(len(edge_data))):
    try:
        user_index = int(edge_data['source_id'][i])
        tweet_index = int(edge_data['target_id'][i]) + len(users)
        if tweet_index < len(tweets):
            user_to_posts[user_index].append(tweets['text'][tweet_index])
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue
    
# dict = {i: [] for i in range(len(users))}

# for i in tqdm(range(len(edge_data))):
#     try:
#         user_index = edge_data['source_id'][i]
#         dict[user_index].append(tweets['text'][edge_data['target_id'][i] + len(users)])
#     except:
#         continue

path3 = '/dev/shm/twi20/data/'
np.save(path3+'user_tweets_dict.npy', dict)
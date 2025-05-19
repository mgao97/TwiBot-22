import os
import json
import torch
os.environ['CUDA_VISIBLE_DEVICE'] = '5'
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import *

pretrained_weights = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(pretrained_weights)
model = T5EncoderModel.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device='cpu', padding=True, truncation=True, max_length=50)

users_tweets = np.load('/dev/shm/twi20/data/user_tweets_dict.npy', allow_pickle=True).tolist()

tweets_tensor = []
for i in tqdm(range(229580)):
    user_tweets_tensor = []
    try:
        for tweet in users_tweets[i]:
            tweet_tensor = torch.tensor(feature_extractor(tweet)).squeeze(0)
            tweet_tensor = torch.mean(tweet_tensor, dim=0) #一个句子
            
            user_tweets_tensor.append(tweet_tensor)
        user_tweets_tensor = torch.mean(torch.stack(user_tweets_tensor), dim=0)
    except:
        user_tweets_tensor = torch.randn(512)
    tweets_tensor.append(user_tweets_tensor)

path3 = '/dev/shm/twi20/data/'
tweets_tensor = torch.stack(tweets_tensor)
torch.save(tweets_tensor, path3+'tweets_tensor_t5.pt')
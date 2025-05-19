import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os

path = '/dev/shm/twi20/data/'
path1 = '/dev/shm/twi20/processed_data/'

user,tweet=fast_merge(dataset="Twibot-20")

user_text=list(user['description'])
tweet_text = [text for text in tweet.text]
each_user_tweets=np.load(path1+'each_user_tweets.npy',allow_pickle=True)

print(type(each_user_tweets))
print(each_user_tweets.shape)
print(len(each_user_tweets))

feature_extract=pipeline('feature-extraction',model='../../../../Twibot-22/roberta-base',tokenizer='roberta-base',device='cpu',padding=True, truncation=True,max_length=50, add_special_tokens = True)

def Des_embbeding():
        print('Running feature1 embedding')
        path2=path1+"des_tensor.pt"
        if not os.path.exists(path2):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path2)
        else:
            des_tensor=torch.load(path2)
        print('Finished')
        return des_tensor

def tweets_embedding():
        print('Running feature2 embedding')
        path2=path1+"tweets_tensor.pt"
        if not os.path.exists(path2):
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
                if len(each_user_tweets[i])==0:
                    total_each_person_tweets=torch.zeros(768)
                else:
                    for j in range(len(each_user_tweets[i])):
                        each_tweet=tweet_text[each_user_tweets[i][j]]
                        if each_tweet is None:
                            total_word_tensor=torch.zeros(768)
                        else:
                            each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                            for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                                if k==0:
                                    total_word_tensor=each_word_tensor
                                else:
                                    total_word_tensor+=each_word_tensor
                            total_word_tensor/=each_tweet_tensor.shape[1]
                        if j==0:
                            total_each_person_tweets=total_word_tensor
                        elif j==20:
                            break
                        else:
                            total_each_person_tweets+=total_word_tensor
                    if (j==20):
                        total_each_person_tweets/=20
                    else:
                        total_each_person_tweets/=len(each_user_tweets[i])
                        
                tweets_list.append(total_each_person_tweets)
                        
            tweet_tensor=torch.stack(tweets_list)
            torch.save(tweet_tensor,path2+"tweets_tensor.pt")
            
        else:
            tweets_tensor=torch.load(path2)
        print('Finished')

# Des_embbeding()
# print('des_tensor.shape:',Des_embbeding().shape)
tweets_embedding()
print('tweets_tensor.shape:',tweets_embedding().shape)

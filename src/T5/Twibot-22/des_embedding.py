# import torch
# from tqdm import tqdm
# from dataset_tool import fast_merge
# import numpy as np
# from transformers import pipeline, T5Tokenizer, T5EncoderModel
# import os

# user,tweet=fast_merge(dataset="Twibot-20")

# user_text=list(user['description'])
# tweet_text = [text for text in tweet.text]

# pretrained_weights = 't5-small'
# model = T5EncoderModel.from_pretrained(pretrained_weights)
# tokenizer = T5Tokenizer.from_pretrained(pretrained_weights)
# feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer,device=2,padding=True, truncation=True,max_length=50)

# def Des_embbeding():
#         print('Running feature1 embedding')
#         path="data/T5/Twibot-20/des_tensor.pt"
#         if not os.path.exists(path):
#             des_vec=[]
#             for k,each in enumerate(tqdm(user_text)):
#                 if each is None:
#                     des_vec.append(torch.zeros(512))
#                 else:
#                     feature=torch.Tensor(feature_extract(each))
#                     for (i,tensor) in enumerate(feature[0]):
#                         if i==0:
#                             feature_tensor=tensor
#                         else:
#                             feature_tensor+=tensor
#                     feature_tensor/=feature.shape[1]
#                     des_vec.append(feature_tensor)
                    
#             des_tensor=torch.stack(des_vec,0)
#             torch.save(des_tensor,path)
#         else:
#             des_tensor=torch.load(path)
#         print('Finished')
#         return des_tensor

# def tweets_embedding():
#         print('Running feature2 embedding')
#         each_user_tweets = np.load('src/T5/cresci-2017/user_tweets_dict.npy', allow_pickle=True)
#         path="src/RoBERTa/cresci-2017/tweets_tensor.pt"
#         if not os.path.exists(path):
#             tweets_list=[]
#             for i in tqdm(range(len(user_text))):
#                 if len(each_user_tweets[i])==0:
#                     total_each_person_tweets=torch.zeros(768)
#                 else:
#                     for j in range(len(each_user_tweets[i])):
#                         each_tweet=tweet_text[each_user_tweets[i][j]]
#                         if each_tweet is None:
#                             total_word_tensor=torch.zeros(768)
#                         else:
#                             each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
#                             for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
#                                 if k==0:
#                                     total_word_tensor=each_word_tensor
#                                 else:
#                                     total_word_tensor+=each_word_tensor
#                             total_word_tensor/=each_tweet_tensor.shape[1]
#                         if j==0:
#                             total_each_person_tweets=total_word_tensor
#                         elif j==20:
#                             break
#                         else:
#                             total_each_person_tweets+=total_word_tensor
#                     if (j==20):
#                         total_each_person_tweets/=20
#                     else:
#                         total_each_person_tweets/=len(each_user_tweets[i])
                        
#                 tweets_list.append(total_each_person_tweets)
                        
#             tweet_tensor=torch.stack(tweets_list)
#             torch.save(tweet_tensor,"./processed_data/tweets_tensor.pt")
            
#         else:
#             tweets_tensor=torch.load(path)
#         print('Finished')

# Des_embbeding()
# tweets_embedding()




import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import os

# 数据加载
import torch
from tqdm import tqdm
from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json

import pickle as pickle  # or import dill as pickle



user=pd.read_json('/dev/shm/twi22/data/user.json')

user_text=list(user['description'])

# 模型加载
pretrained_weights = 't5-small'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(pretrained_weights)
model = T5EncoderModel.from_pretrained(pretrained_weights).to(device)
model.eval()

# 嵌入函数
def Des_embedding(batch_size=1024, max_length=50):
    print('Running feature1 embedding')
    path = "/dev/shm/twi22/data/des_tensor_t5.pt"
    
    if os.path.exists(path):
        print('Loading cached embeddings...')
        return torch.load(path)
    
    des_vec = []

    with torch.no_grad():
        for i in tqdm(range(0, len(user_text), batch_size)):
            batch_texts = user_text[i:i+batch_size]
            
            # 替换 None 为 ""
            batch_texts = [" " if text is None else text for text in batch_texts]
            
            # Tokenize
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Run model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_size)
            
            # 取平均（忽略 padding）
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
            summed = torch.sum(embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            averaged = summed / counts
            
            des_vec.append(averaged.cpu())
    
    des_tensor = torch.cat(des_vec, dim=0)
    torch.save(des_tensor, path)
    print('Finished')
    return des_tensor

des_tensor = Des_embedding()
print(des_tensor.shape)  # e.g. torch.Size([num_users, 512])


from cProfile import label
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path

def str_to_int(s):
    if s == 'human':
        return 1
    else:
        return 0

path1 = '/dev/shm/twi20/data'
data = pd.read_csv(path1+'/label.csv')

data_label = {}
for id in data['id']:
    data_label[id] = str_to_int(data['label'][data['id'] == id].item())

path2 = '/dev/shm/twi20/data'
with open(path2+'/id_list.json', 'r') as f:
    id_list = json.loads(f.read())

label_list = []
for id in id_list:
    try:
        label_list.append(data_label[id])
    except:
        label_list.append(-1)
torch.save(torch.tensor(label_list), path2+'/label_list.pt')


"""
test part
"""
data = torch.load(path2+'/label_list.pt')
for i, item in enumerate(data):
    if item == -1:
        print(i)
        break
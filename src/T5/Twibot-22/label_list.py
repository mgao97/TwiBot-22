from cProfile import label
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path

# def str_to_int(s):
#     if s == 'human':
#         return 1
#     else:
#         return 0

# path1 = Path('/dev/shm/twi22/data')
# data = pd.read_csv(path1 / 'label.csv')

# data_label = {}
# for id in data['id']:
#     data_label[id] = str_to_int(data['label'][data['id'] == id].item())

# # path2 = Path('src/T5/Twibot-20')
# with open(path1 / 'id_list.json', 'r') as f:
#     id_list = json.loads(f.read())

# label_list = []
# for id in id_list:
#     try:
#         label_list.append(data_label[id])
#     except:
#         label_list.append(-1)
# torch.save(torch.tensor(label_list), path1 / 'label_list.pt')


# """
# test part
# """
# data = torch.load(path1 / 'label_list.pt')
# for i, item in enumerate(data):
#     if item == -1:
#         print(i)
#         break


import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path

def str_to_int(s):
    return 1 if s == 'human' else 0

path1 = Path('/dev/shm/twi22/data')

# 读取数据，利用map + vectorize替代for循环
data = pd.read_csv(path1 / 'label.csv')

# 直接映射label列为0/1
data['label_int'] = data['label'].map(str_to_int)

# 建立id -> label_int映射dict
data_label = dict(zip(data['id'], data['label_int']))

# 读取id_list.json
with open(path1 / 'id_list.json', 'r') as f:
    id_list = json.load(f)

# 用dict.get避免异常，找不到的返回-1
label_list = [data_label.get(id, -1) for id in id_list]

# 保存tensor
torch.save(torch.tensor(label_list), path1 / 'label_list.pt')


"""
test part
"""
data = torch.load(path1 / 'label_list.pt')
for i, item in enumerate(data):
    if item == -1:
        print(i)
        break

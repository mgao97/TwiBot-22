from model import BotRGCN
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score,precision_recall_curve

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import argparse
import os
import numpy as np
import random

parser = argparse.ArgumentParser(description='BotRGCN')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

experiment_seed = args.seed
torch.manual_seed(experiment_seed)
torch.cuda.manual_seed(experiment_seed)

device = 'cpu'
embedding_size,dropout,lr,weight_decay=32,0.1,1e-2,5e-2


root='/dev/shm/twi22/processed_data/'

dataset=Twibot22(root=root,device=device,process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()


model=BotRGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)


def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test():
    model.eval()
    results = []
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    pred_binary = torch.argmax(output, dim=1)

    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    #mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    precision=precision_score(label[test_idx],output[test_idx])
    recall=recall_score(label[test_idx],output[test_idx])
    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc=auc(fpr, tpr)

    # 计算 Precision-Recall 曲线
    precision1, recall1, _ = precision_recall_curve(label[test_idx], pred_binary[test_idx].cpu().numpy())  # 使用正类（label=1）的预测概率
    # 计算 AUCPR
    aucpr = auc(recall1, precision1)

    # 计算精确率为80%时的召回率
    recall_at_p80 = 0
    for pi, ri in zip(precision1, recall1):
        if pi >= 0.8:
            recall_at_p80 = ri
            break

    # 计算精确率为85%时的召回率
    recall_at_p85 = 0
    for pi, ri in zip(precision1, recall1):
        if pi >= 0.85:
            recall_at_p85 = ri
            break

    # 计算精确率为90%时的召回率
    recall_at_p90 = 0
    for pi, ri in zip(precision1, recall1):
        if pi >= 0.9:
            recall_at_p90 = ri
            break


    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "precision= {:.4f}".format(precision),
            "recall= {:.4f}".format(recall),
            "f1_score= {:.4f}".format(f1),
            "aucroc= {:.4f}".format(Auc),
            "aucpr= {:.4f}".format(aucpr),
            "recall at precision 80%={:.4f}".format(recall_at_p80),
            "recall at precision 85%={:.4f}".format(recall_at_p85),
            "recall at precision 90%={:.4f}".format(recall_at_p90)
            )
    
    # 记录结果
    results.append({
        
        'seed': experiment_seed,
        'Accuracy': acc_test.item(),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUCROC': Auc,
        'AUC-PR': aucpr,
        'Recall@P80': recall_at_p80,
        'Recall@P85': recall_at_p85,
        'Recall@P90': recall_at_p90
    })
    # 生成表格
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))

    # 保存结果到CSV
    results_df.to_csv('BotRGCN.csv', mode='a', header=not os.path.exists(f'BotRGCN.csv'), index=False)
    
model.apply(init_weights)

epochs=200
for epoch in range(epochs):
    train(epoch)
    
test()
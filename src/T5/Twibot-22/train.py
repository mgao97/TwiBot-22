import os
import numpy as np
import pandas as pd
import torch
# os.environ['CUDA_VISIBLE_DEVICE'] = '5'
import math
from tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


import argparse
import os
import pandas as pd
import numpy as np

# CUDA = 'cuda:6'
device = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-22', help='Choose the dataset.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
args = parser.parse_args()

split = [[], [], []]
path0 = '/dev/shm/twi22/data/'
split_list = pd.read_csv(path0+'split.csv')
label = pd.read_csv(path0+'label.csv')

users_index_to_uid = list(label['id'])
uid_to_users_index = {x : i for i, x in enumerate(users_index_to_uid)}
for id in split_list[split_list['split'] == 'train']['id']:
    split[0].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'val']['id']:
    split[1].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'test']['id']:
    split[2].append(uid_to_users_index[id])

def eval(preds_auc, preds, labels):
    print("ACC:{}".format(accuracy_score(labels, preds)), end=",")
    print("F1:{}".format(f1_score(labels, preds)), end=",")
    print("ROC:{}".format(roc_auc_score(labels, preds_auc)))
    print("precision_score:{}".format(precision_score(labels, preds)), end=",")
    print("recall_score:{}".format(recall_score(labels, preds)))

            
class Twibot20Dataset(Dataset):
    def __init__(self, name, device='cpu'):
        self.device = torch.device(device)
        path1 = '/dev/shm/twi22/processed_data/'
        path2 = '/dev/shm/twi22/data/'
        
        
        tweets_tensor = torch.load(path2 +'tweets_tensor_t5.pt')
        des_tensor = torch.load(path2+'des_tensor_t5.pt')
        label = 1 - torch.load(path2+'label_list.pt')
        
        if name == 'train':
            self.tweet_feature = tweets_tensor[split[0]]
            self.des_feature = des_tensor[split[0]]
            self.label = label[split[0]]
            self.length = len(self.tweet_feature)
        elif name == 'val':
            self.tweet_feature = tweets_tensor[split[1]]
            self.des_feature = des_tensor[split[1]]
            self.label = label[split[1]]
            self.length = len(self.tweet_feature)
        else:
            self.tweet_feature = tweets_tensor[split[2]]
            self.des_feature = des_tensor[split[2]]
            self.label = label[split[2]]
            self.length = len(self.tweet_feature)
        """
        batch_size here is useless
        """
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.tweet_feature[index], self.des_feature[index], self.label[index]
    
    
class MLPclassifier(nn.Module):
    def __init__(self,
                 input_dim=512,
                 hidden_dim=128,
                 dropout=0.5):
        super(MLPclassifier, self).__init__()
        self.dropout = dropout
        
        self.pre_model1 = nn.Linear(input_dim, input_dim // 2)
        self.pre_model2 = nn.Linear(input_dim, input_dim // 2)
        
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self,tweet_feature, des_feature):
        pre1 = self.pre_model1(tweet_feature)
        pre2 = self.pre_model2(des_feature)
        x = torch.cat((pre1,pre2), dim=1)
        x = self.linear_relu_tweet(x)
        # x = self.linear_relu1(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class RobertaTrianer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 epochs=50,
                 input_dim=512,
                 hidden_dim=128,
                 dropout=0.5,
                 activation='relu',
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-5,
                 lr=1e-4,
                 device='cpu'):    
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        self.device = device
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self):
        train_loader = self.train_loader
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            preds = []
            preds_auc = []
            labels = []
            with tqdm(train_loader) as progress_bar:
                for batch in progress_bar:
                    tweet = batch[0].to(self.device)
                    des = batch[1].to(self.device)
                    label = batch[2].to(self.device)
                    pred = self.model(tweet, des)
                    loss = self.loss_func(pred, label)
                    loss_avg += loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())
                    
                    preds.append(pred.argmax(dim=-1).cpu().numpy())
                    preds_auc.append(pred[:,1].detach().cpu().numpy())
                    labels.append(label.cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            preds_auc = np.concatenate(preds_auc, axis=0)
            labels = np.concatenate(labels, axis=0)
            loss_avg = loss_avg / len(train_loader)   
            print('{' + f'loss={loss_avg.item()}' + '}' + 'eval=', end='')
            eval(preds_auc, preds, labels)     
            self.valid()
            self.test()
        
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        val_loader = self.val_loader
        for batch in val_loader:
            tweet = batch[0].to(self.device)
            des = batch[1].to(self.device)
            label = batch[2].to(self.device)
            pred = self.model(tweet, des)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        

        preds = np.concatenate(preds, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        eval(preds_auc, preds, labels)
    
    def test(self):
        self.model.eval()
        preds = []
        probs = []
        preds_auc = []
        labels = []
        acc = []
        precision = []
        recall = []
        f1 = []
        aucpr = []
        results = []

        test_loader = self.test_loader
        for batch in test_loader:
            pred = self.model(batch[0], batch[1])
            prob = pred[:, 1].detach().cpu().numpy()

            preds.append(pred.argmax(dim=-1).cpu().numpy())
            probs.append(prob)

            acc.append(accuracy_score(batch[2].cpu().numpy(), pred.argmax(dim=-1).cpu().numpy()))
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(batch[2].cpu().numpy())
            precision.append(precision_score(batch[2].cpu().numpy(), pred.argmax(dim=-1).cpu().numpy()))
            recall.append(recall_score(batch[2].cpu().numpy(), pred.argmax(dim=-1).cpu().numpy()))
            f1.append(f1_score(batch[2].cpu().numpy(), pred.argmax(dim=-1).cpu().numpy()))
            aucpr.append(roc_auc_score(batch[2].cpu().numpy(), pred[:,1].detach().cpu().numpy()))

            # precision1, recall1, thresholds = precision_recall_curve(batch[2].cpu().numpy(), probs)
            # aucpr.append(auc(recall1, precision1))

        # 把所有batch的标签和概率拼接成一个整体
        all_labels = np.concatenate(labels)
        all_probs = np.concatenate(probs)

        precision1, recall1, thresholds = precision_recall_curve(all_labels, all_probs)
        aucpr.append(auc(recall1, precision1))

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


        acc = np.mean(acc)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        aucpr = np.mean(aucpr)
        preds_auc = np.concatenate(preds_auc)
        preds = np.concatenate(preds)


                # 记录结果
        results.append({
            'seed': args.seed,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUCROC': np.mean(preds_auc),
            'AUC-PR': aucpr,
            'Recall@P80': recall_at_p80,
            'Recall@P85': recall_at_p85,
            'Recall@P90': recall_at_p90
        })
        # 生成表格
        results_df = pd.DataFrame(results)
        print(results_df.to_markdown(index=False))

        # 保存结果到CSV
        results_df.to_csv('T5_twi22.csv', mode='a', header=not os.path.exists(f'T5_twi22.csv'), index=False)
        eval(preds_auc, preds, labels)
    # @torch.no_grad()
    # def test(self):
    #     self.model.eval()
    #     preds = []
    #     preds_auc = []
    #     labels = []
    #     test_loader = self.test_loader
    #     for batch in test_loader:
    #         tweet = batch[0].to(self.device)
    #         des = batch[1].to(self.device)
    #         label = batch[2].to(self.device)
    #         pred = self.model(tweet, des)
    #         preds.append(pred.argmax(dim=-1).cpu().numpy())
    #         preds_auc.append(pred[:,1].detach().cpu().numpy())
    #         labels.append(label.cpu().numpy())
            
    #     preds = np.concatenate(preds, axis=0)
    #     preds_auc = np.concatenate(preds_auc, axis=0)
    #     labels = np.concatenate(labels, axis=0)
        
    #     eval(preds_auc, preds, labels)
  
     
        
if __name__ == '__main__':
    train_dataset = Twibot20Dataset('train')
    val_dataset = Twibot20Dataset('val')
    test_dataset = Twibot20Dataset('test')
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # for i in range(5):
    trainer = RobertaTrianer(train_loader, val_loader, test_loader)
    trainer.train()
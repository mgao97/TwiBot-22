# from Dataset import Twibot22
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,precision_recall_curve, auc
# from layer import RGTLayer
import pytorch_lightning as pl
from torch import nn
import torch
# from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler

from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F

import dhg
from dhg.nn import HGNNConv,UniGINConv, UniSAGEConv

from mulltiattn import MultiAttnModel

import numpy as np
from MGTAB import MGTABDataset

import os
from pytorch_lightning.loggers import TensorBoardLogger
import gc
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import argparse
import torch.nn.functional as F


def relabel_edges(edge_index):
    # 提取 edge_index 中所有唯一的节点 ID
    all_nodes = torch.cat([edge_index[0], edge_index[1]], dim=0)
    unique_nodes = torch.unique(all_nodes)
    
    # 创建一个字典，将每个节点 ID 映射到一个新的连续 ID
    node_to_new_id = {old_id.item(): new_id for new_id, old_id in enumerate(unique_nodes)}
    
    # 使用映射来重新编码 edge_index
    edge_index = edge_index.t()  # 转置 edge_index
    for i in range(edge_index.size(0)):
        edge_index[i] = torch.tensor([node_to_new_id[node.item()] for node in edge_index[i]])  # 映射到新的 ID
    edge_index = edge_index.t()  # 再次转置回原来的形态
    
    return edge_index

def load_data(args):
    print("loading features...")
    x = torch.load(args.path + "features.pt", map_location="cpu")
    print('x.shape:',x.shape)
    
    print("loading edges & label...")
    edge_index = torch.load(args.path + "edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "edge_type.pt", map_location="cpu").unsqueeze(-1)
    # edge_weight = torch.load(args.path + "edge_weight.pt", map_location="cpu")
    label = torch.load(args.path + "labels_bot.pt", map_location="cpu")
    data = Data(x=x, edge_index = edge_index, edge_type=edge_type, label=label)
    node_num = x.shape[0]
    sample_idx = shuffle(np.array(range(node_num)),
                                 random_state=args.seed)
    data.train_idx = sample_idx[:int(0.7 * node_num)]
    data.valid_idx = sample_idx[int(0.7 * node_num):int(0.9 * node_num)]
    data.test_idx = sample_idx[int(0.9 * node_num):]
    return data


class BotHybrid(pl.LightningModule):
    def __init__(self,args):
        super(BotHybrid, self).__init__()

        self.lr = args.lr
        self.l2_reg = args.l2_reg
        self.dropout = args.dropout
        self.k = args.k
        
        # self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        # self.CELoss = CustomCrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.relu = nn.LeakyReLU()

        self.linear_relu_des=nn.Sequential(
            nn.Linear(args.des_channel,int(args.hid_channels/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(args.tweet_channel,int(args.hid_channels/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(args.numeric_num,int(args.hid_channels/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(args.cat_num,int(args.hid_channels/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_feats = nn.Sequential(
            nn.Linear(args.hid_channels, args.hid_channels),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(788,args.hid_channels),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(args.hid_channels,args.hid_channels,num_relations=7)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(args.hid_channels,int(args.hid_channels/2)),
            nn.LeakyReLU()
        )
        self.HGNN_layer1 = HGNNConv(in_channels=int(args.hid_channels/2)+args.hid_channels, out_channels=args.hid_channels, use_bn=args.use_bn, drop_rate=args.dropout)
        self.HGNN_layer2 = HGNNConv(in_channels=args.hid_channels, out_channels=int(args.hid_channels/2), use_bn=args.use_bn, is_last=True)
        # self.HGNN_layer1 = UniSAGEConv(in_channels=int(args.hid_channels/2)+args.hid_channels, out_channels=args.hid_channels, use_bn=args.use_bn, drop_rate=args.dropout)
        # self.HGNN_layer2 = UniSAGEConv(in_channels=args.hid_channels, out_channels=int(args.hid_channels/2), use_bn=args.use_bn, is_last=True)
        self.multiattn = MultiAttnModel(num_layers=args.num_layers, model_dim=int(args.hid_channels/2), num_heads=args.num_heads, hidden_dim=args.hid_channels, dropout_rate=args.dropout)
        
        self.linear_output2=nn.Linear(args.hid_channels,2)
        
        self.init_weight()



    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def training_step(self,train_batch,batch_idx):
        # cat_prop = train_batch.x[:, :args.cat_num]
        # num_prop = train_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
        # tweet = train_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
        # des = train_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_type.view(-1)
        label = train_batch.label
        
        # d=self.linear_relu_des(des)
        # t=self.linear_relu_tweet(tweet)
        # n=self.linear_relu_num_prop(num_prop)
        # c=self.linear_relu_cat_prop(cat_prop)
        # x_in=torch.cat((d,t,n,c),dim=1)

        x_in = train_batch.x
        x=self.linear_relu_input(x_in)

        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x_low=self.linear_relu_output1(x)
        
        x_new = torch.cat((x_low,x_in),dim=1)
        
        hg = dhg.Hypergraph.from_feature_kNN(x_new.float().detach(), k=self.k)
        x_new = x_new.to(device)
        hg = hg.to(device)

        x=self.HGNN_layer1(x_new,hg)
        x_high=self.HGNN_layer2(x,hg)
        # fused_x = torch.cat((x_low,x_high),dim=-1)
        fused_low, fused_high = self.multiattn(x_low.unsqueeze(1), x_high.unsqueeze(1))

        fused_x = torch.cat((fused_low.squeeze(1),fused_high.squeeze(1)),dim=-1)

        fused_x = self.relu(fused_x)

        pred=self.linear_output2(fused_x)
        loss = self.CELoss(pred, label)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # print('val batch:',val_batch)
        # print('='*100)
        self.eval()
        with torch.no_grad():
            x_in = val_batch.x
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_type.view(-1)
            label = val_batch.label
            
            x=self.linear_relu_input(x_in)
            x=self.rgcn(x,edge_index,edge_type)
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.rgcn(x,edge_index,edge_type)
            x_low=self.linear_relu_output1(x)

            x_new = torch.cat((x_low,x_in),dim=1)
            
            hg = dhg.Hypergraph.from_feature_kNN(x_new.float().detach(), k=self.k)
            x_new = x_new.to(device)
            hg = hg.to(device)
            # x = x.to('cpu')
            # hg = hg.to('cpu')

            x=self.HGNN_layer1(x_new,hg)
            x_high=self.HGNN_layer2(x,hg)

            # fused_x = torch.cat((x_low,x_high),dim=-1)
            fused_low, fused_high = self.multiattn(x_low.unsqueeze(1), x_high.unsqueeze(1))

            fused_x = torch.cat((fused_low.squeeze(1),fused_high.squeeze(1)),dim=-1)

            fused_x = self.relu(fused_x)

            pred=self.linear_output2(fused_x)
            loss = self.CELoss(pred, label)
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            # rocauc = roc_auc_score(label.cpu(), pred[:,1].cpu())
            
            self.log("val_acc", acc, prog_bar=True, batch_size=label.size(0))
            self.log("val_f1", f1, prog_bar=True, batch_size=label.size(0))
            self.log("val_loss", loss)
            # self.log("val_rocauc", rocauc)
            
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            x_in = test_batch.x
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_type.view(-1)
            label = test_batch.label
            
            x=self.linear_relu_input(x_in)
            x=self.rgcn(x,edge_index,edge_type)
            x=F.dropout(x,p=self.dropout,training=self.training)
            x=self.rgcn(x,edge_index,edge_type)
            x_low=self.linear_relu_output1(x)
            x_new = torch.cat((x_low,x_in),dim=1)
            hg = dhg.Hypergraph.from_feature_kNN(x_new.float().detach(), k=self.k)
            x_new = x_new.to(device)
            hg = hg.to(device)

            x=self.HGNN_layer1(x_new,hg)
            x_high=self.HGNN_layer2(x,hg)

            fused_low, fused_high = self.multiattn(x_low.unsqueeze(1), x_high.unsqueeze(1))

            fused_x = torch.cat((fused_low.squeeze(1),fused_high.squeeze(1)),dim=-1)

            fused_x = self.relu(fused_x)

            pred=self.linear_output2(fused_x)

            # torch.save(fused_x,'twibot22_emb.pt')
            # torch.save(label,'twibot22_y.pt')

            pred_binary = torch.argmax(pred, dim=1)

            label_np = label.detach().cpu().float().numpy()
            pred_prob_np = pred[:, 1].detach().cpu().float().numpy()
            Auc = roc_auc_score(label_np, pred_prob_np)


            pred_binary_np = pred_binary.detach().cpu().float().numpy()
            acc = accuracy_score(label_np, pred_binary_np)
            f1 = f1_score(label_np, pred_binary_np)
            precision = precision_score(label_np, pred_binary_np)
            recall = recall_score(label_np, pred_binary_np)

            # 计算 Precision-Recall 曲线
            precision1, recall1, _ = precision_recall_curve(label_np, pred_prob_np)

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
                    "test_accuracy= {:.4f}".format(acc),
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
                'seed': args.seed,
                'Accuracy': acc,
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
            results_df.to_csv('HyperScan_mgtab.csv', mode='a', header=not os.path.exists(f'HyperScan_mgtab.csv'), index=False)

            
            # pred_test.append(pred_binary.squeeze().cpu())
            # pred_test_prob.append(pred[:,1].squeeze().cpu())
            # label_test.append(label.squeeze().cpu())

            # print(pred_binary.shape,label.shape)

            # acc = accuracy_score(label.cpu(), pred_binary.cpu())
            # f1 = f1_score(label.cpu(), pred_binary.cpu())
            # precision =precision_score(label.cpu(), pred_binary.cpu())
            # recall = recall_score(label.cpu(), pred_binary.cpu())
            # rocauc = roc_auc_score(label.cpu(), pred[:,1].cpu())

            # # 计算 Precision-Recall 曲线
            # precision1, recall1, _ = precision_recall_curve(label.cpu(), pred[:, 1].cpu())  # 使用正类（label=1）的预测概率

            # # 计算 AUCPR
            # aucpr = auc(recall1, precision1)

            

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("rocauc", Auc)
            self.log("aucpr", aucpr)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }
    




parser = argparse.ArgumentParser(description="HyperScan")
parser.add_argument("--path", type=str, default="/dev/shm/mgtab/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=10, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=10, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=8, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--hid_channels", type=int, default=788, help="hidden channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--trans_head", type=int, default=2, help="description channel")
parser.add_argument("--semantic_head", type=int, default=2, help="description channel")
parser.add_argument("--batch_size", type=int, default=512, help="description channel")
parser.add_argument("--epochs", type=int, default=200, help="description channel")
parser.add_argument("--lr", type=float, default=0.001, help="description channel")
parser.add_argument("--l2_reg", type=float, default=1e-5, help="description channel")
parser.add_argument("--seed", type=int, default=0, help="random")
parser.add_argument("--test_batch_size", type=int, default=200, help="random")
parser.add_argument("--use_bn",action='store_true',default=True)
parser.add_argument("--num_layers",type=int, default=4, help="num layers")
parser.add_argument("--num_heads",type=int, default=4, help="num heads")
parser.add_argument("--k",default=8,type=int)

if __name__ == "__main__":
    
    global results
    args = parser.parse_args()
    results = []
   
       
    if args.seed != None:
        pl.seed_everything(args.seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='max'
    )
    
    data = load_data(args)
    print('MGTAB data:',data)
    

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    data = data.to(device)
    
    print("loading...")
    gc.collect()  # 手动触发垃圾回收
    train_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.train_idx, batch_size=args.batch_size, num_workers=20,shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.valid_idx, batch_size=args.batch_size, num_workers=20)
    test_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.test_idx, batch_size=args.test_batch_size, num_workers=20)

    model=BotHybrid(args)
    print('args:',args)

    model = model.to(device)
    print('model:',model)

    trainer = pl.Trainer(num_nodes=1, max_epochs=args.epochs, precision=32, log_every_n_steps=1,
                        callbacks=[early_stop_callback,checkpoint_callback])
    trainer.fit(model, train_loader, valid_loader,ckpt_path='./lightning_logs/version_0/checkpoints/val_acc=0.8587.ckpt')
    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_model = BotHybrid.load_from_checkpoint(checkpoint_path=best_path, args=args,strict=False)
    trainer.test(best_model, test_loader, verbose=True)
    
    
    
    
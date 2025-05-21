from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,precision_recall_curve,auc
from torch_geometric.nn import HGTConv
import pytorch_lightning as pl
from torch import nn
import torch
from Dataset import BotDataset
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
import os
import pandas as pd
import numpy as np


class HGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(HGTDetector, self).__init__()
        edge_index = torch.load(args.path + "edge_index.pt", map_location="cpu")
        edge_type = torch.load(args.path + "edge_type.pt", map_location="cpu").long()
        
        self.following_edge_index = edge_index[:, edge_type==0]
        self.follower_edge_index = edge_index[:, edge_type==1]
        self.label = torch.load(args.path + "label.pt", map_location="cpu")
  
        self.lr = args.lr
        self.l2_reg = args.l2_reg

        self.cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cpu")
        self.prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cpu")
        self.tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cpu")
        self.des_features = torch.load(args.path + "des_tensor.pt", map_location="cpu")

        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.HGT_layer1 = HGTConv(in_channels=args.linear_channels, out_channels=args.linear_channels, metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
        self.HGT_layer2 = HGTConv(in_channels=args.linear_channels, out_channels=args.linear_channels, metadata=(['user'], [('user', 'follower', 'user'), ('user', 'following', 'user')]))
        
        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)
        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def training_step(self, train_batch, batch_idx):
        train_batch = train_batch.squeeze(0)

        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        x_dict = {"user":user_features}
        edge_index_dict = {('user','follower','user'): self.follower_edge_index,
                           ('user','following','user'): self.following_edge_index}
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer2(x_dict, edge_index_dict)

        user_features = self.drop(self.ReLU(self.out1(x_dict["user"])))
        pred = self.out2(user_features[train_batch])
        loss = self.CELoss(pred, self.label[train_batch])

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            val_batch = val_batch.squeeze(0)

            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            x_dict = {"user":user_features}
            edge_index_dict = {('user','follower','user'): self.follower_edge_index,
                            ('user','following','user'): self.following_edge_index}
            x_dict = self.HGT_layer1(x_dict, edge_index_dict)
            x_dict = self.HGT_layer2(x_dict, edge_index_dict)

            user_features = self.drop(self.ReLU(self.out1(x_dict["user"])))
            pred = self.out2(user_features[val_batch])
            # print(pred.size())
            pred_binary = torch.argmax(pred, dim=1)
            
            acc = accuracy_score(self.label[val_batch].cpu(), pred_binary.cpu())
            f1 = f1_score(self.label[val_batch].cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc,on_step=False, batch_size=self.label.size(0))
            self.log("val_f1", f1,on_step=False, batch_size=self.label.size(0))

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            test_batch = test_batch.squeeze(0)
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(self.prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(self.cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(self.tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(self.des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            x_dict = {"user":user_features}
            edge_index_dict = {('user','follower','user'): self.follower_edge_index,
                            ('user','following','user'): self.following_edge_index}
            x_dict = self.HGT_layer1(x_dict, edge_index_dict)
            x_dict = self.HGT_layer2(x_dict, edge_index_dict)

            user_features = self.drop(self.ReLU(self.out1(x_dict["user"])))
            torch.save(user_features[test_batch], "HGT_twi20_embedding.pt")
            pred = self.out2(user_features[test_batch])
            pred_binary = torch.argmax(pred, dim=1)

            label_np = self.label[test_batch].detach().cpu().float().numpy()
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
            results_df.to_csv('HGT_twi20.csv', mode='a', header=not os.path.exists(f'HGT_twi20.csv'), index=False)

            # acc = accuracy_score(self.label[test_batch].cpu(), pred_binary.cpu())
            # f1 = f1_score(self.label[test_batch].cpu(), pred_binary.cpu())
            # precision =precision_score(self.label[test_batch].cpu(), pred_binary.cpu())
            # recall = recall_score(self.label[test_batch].cpu(), pred_binary.cpu())
            # auc = roc_auc_score(self.label[test_batch].cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", Auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, Auc))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="/dev/shm/twi20/processed_data/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--rel_dim", type=int, default=100, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--batch_size", type=int, default=128, help="description channel")
parser.add_argument("--epochs", type=int, default=50, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--seed", type=int, default=None, help="random")
parser.add_argument("--beta", type=float, default=0.05, help="description channel")

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    global results
    results = []

    if args.seed != None:
        pl.seed_everything(args.seed)
        
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=1,
        verbose=True)

    train_dataset = BotDataset(name="train")
    valid_dataset = BotDataset(name="valid")
    test_dataset = BotDataset(name="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = HGTDetector(args)
    trainer = pl.Trainer(num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1, callbacks=[checkpoint_callback])
    
    # trainer.fit(model, train_loader, valid_loader)

    # dir = './hgt/lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    # best_path = './hgt/lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_path = './hgt/lightning_logs/version_0/checkpoints/val_acc=0.8719.ckpt'

    best_model = HGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)

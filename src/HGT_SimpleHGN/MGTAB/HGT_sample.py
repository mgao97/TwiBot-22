from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,precision_recall_curve,auc
from torch_geometric.nn import HGTConv
import pytorch_lightning as pl
from torch import nn
import torch
from torch_geometric.loader import HGTLoader, NeighborLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from os import listdir
from torch_geometric.data import HeteroData
import os
import pandas as pd
import numpy as np

def load_data(args):
    
    print("loading features...")
    # cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cpu")
    # prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cpu")
    # tweet_features = torch.load(args.path + "user_tweet_feats.pt", map_location="cpu")
    # des_features = torch.load(args.path + "user_des_feats1.pt", map_location="cpu")
    # x = torch.cat((cat_features, prop_features, tweet_features, des_features), dim=1)
    x = torch.load(args.path + "features.pt", map_location="cpu")
    
    print("loading edges & label...")
    edge_index = torch.load(args.path + "edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "edge_type.pt", map_location="cpu")
    label = torch.load(args.path + "labels_bot.pt", map_location="cpu")
    
    data = HeteroData()
    data["user"].x = x
    data["user"].y = label
    data["user", "follower", "user"].edge_index = edge_index[:, edge_type==0]
    data["user", "following", "user"].edge_index = edge_index[:, edge_type==1]
    
    print("loading index...")
    data.train_idx = torch.load(args.path + "train_idx.pt", map_location="cpu")
    data.valid_idx = torch.load(args.path + "val_idx.pt", map_location="cpu")
    data.test_idx = torch.load(args.path + "test_idx.pt", map_location="cpu")
    
    return data

class HGTDetector(pl.LightningModule):
    def __init__(self, args):
        super(HGTDetector, self).__init__()
        # self.edge_index = torch.load(args.path + "edge_index.pt", map_location="cuda")
        # self.edge_type = torch.load(args.path + "edge_type.pt", map_location="cuda").long()
        # self.label = torch.load(args.path + "label.pt", map_location="cuda")
  
        self.lr = args.lr
        self.l2_reg = args.l2_reg

        # self.cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cuda")
        # self.prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cuda")
        # self.tweet_features = torch.load(args.path + "tweets_tensor.pt", map_location="cuda")
        # self.des_features = torch.load(args.path + "des_tensor.pt", map_location="cuda")

        # self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        # self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        # self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        # self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(788, args.linear_channels)

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
        # cat_features = train_batch["user"].x[:, :args.cat_num]
        # prop_features = train_batch["user"].x[:, args.cat_num: args.cat_num + args.numeric_num]
        # tweet_features = train_batch["user"].x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
        # des_features = train_batch["user"].x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
        
        user_features = train_batch.x
        label = train_batch["user"].y
        # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))
        
        x_dict = {"user":user_features}
        x_dict = self.HGT_layer1(x_dict, train_batch.edge_index_dict)        

        x_dict = self.HGT_layer2(x_dict, train_batch.edge_index_dict)

        user_features = self.ReLU(self.out1(x_dict["user"]))
        pred = self.out2(user_features)
        loss = self.CELoss(pred, label)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            # cat_features = val_batch["user"].x[:, :args.cat_num]
            # prop_features = val_batch["user"].x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = val_batch["user"].x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = val_batch["user"].x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]

            user_features = val_batch.x
            label = val_batch["user"].y
            # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))
            
            x_dict = {"user":user_features}
            x_dict = self.HGT_layer1(x_dict, val_batch.edge_index_dict)        

            x_dict = self.HGT_layer2(x_dict, val_batch.edge_index_dict)

            user_features = self.ReLU(self.out1(x_dict["user"]))
            pred = self.out2(user_features)

            pred_binary = torch.argmax(pred, dim=1)
            # print(label.size())
            # print(pred_binary.size())
            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc, on_step=False, batch_size=label.size(0))
            self.log("val_f1", f1, on_step=False, batch_size=label.size(0))

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            # cat_features = test_batch["user"].x[:, :args.cat_num]
            # prop_features = test_batch["user"].x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = test_batch["user"].x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = test_batch["user"].x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]

            user_features = test_batch.x
            label = test_batch["user"].y[:args.test_batch_size]
            # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))
            
            x_dict = {"user":user_features}
            x_dict = self.HGT_layer1(x_dict, test_batch.edge_index_dict)        

            x_dict = self.HGT_layer2(x_dict, test_batch.edge_index_dict)

            # pred = self.out1(x_dict["user"])
            
            user_features = self.ReLU(self.out1(x_dict["user"]))
            pred = self.out2(user_features)[:args.test_batch_size]
        
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
            results_df.to_csv('HGT_mgtab.csv', mode='a', header=not os.path.exists(f'HGT_mgtab.csv'), index=False)


            # pred_test.append(pred_binary.squeeze().cpu())
            # pred_test_prob.append(pred[:,1].squeeze().cpu())
            # label_test.append(label.squeeze().cpu())
            
            # acc = accuracy_score(label.cpu(), pred_binary.cpu())
            # f1 = f1_score(label.cpu(), pred_binary.cpu())
            # precision =precision_score(label.cpu(), pred_binary.cpu())
            # recall = recall_score(label.cpu(), pred_binary.cpu())
            # auc = roc_auc_score(label.cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", Auc)

            # print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="HGT")
parser.add_argument("--path", type=str, default="/dev/shm/mgtab/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--rel_dim", type=int, default=100, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--batch_size", type=int, default=1024, help="description channel")
parser.add_argument("--epochs", type=int, default=200, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--seed", type=int, default=None, help="random")
parser.add_argument("--test_batch_size", type=int, default=200, help="random")

if __name__ == "__main__":
    global args
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

    # 添加早停回调，连续10个epoch没有提升则停止训练
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='max'
    )

    data = load_data(args)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    # train_loader = HGTLoader(data, num_samples={key: [512] * 2 for key in data.node_types}, input_nodes=("user", data.train_idx), batch_size=args.batch_size, shuffle=True)
    # valid_loader = HGTLoader(data, num_samples={key: [512] * 2 for key in data.node_types}, input_nodes=("user", data.valid_idx), batch_size=args.batch_size)
    # test_loader = HGTLoader(data, num_samples={key: [200] * 2 for key in data.node_types}, input_nodes=("user", data.test_idx), batch_size=args.test_batch_size)

    train_loader = NeighborLoader(data, num_neighbors=[256]*2, input_nodes=("user",data.train_idx), batch_size=args.batch_size, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[256]*2, input_nodes=("user",data.valid_idx), batch_size=args.batch_size)
    test_loader = NeighborLoader(data, num_neighbors=[256]*2, input_nodes=("user",data.test_idx), batch_size=args.test_batch_size)
    
    model = HGTDetector(args)
    trainer = pl.Trainer(num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1, callbacks=[checkpoint_callback, early_stop_callback])
    
    trainer.fit(model, train_loader, valid_loader)


    dir = './lightning_logs/hgt/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path = './lightning_logs/hgt/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])

    best_model = HGTDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,precision_recall_curve,auc
from layer import SimpleHGN
import pytorch_lightning as pl
from torch import nn
import torch
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def load_data(args):
    
    print("loading features...")
    # cat_features = torch.load(args.path + "cat_properties_tensor.pt", map_location="cpu")
    # prop_features = torch.load(args.path + "num_properties_tensor.pt", map_location="cpu")
    # tweet_features = torch.load(args.path + "user_tweet_feats.pt", map_location="cpu")
    # des_features = torch.load(args.path + "user_des_feats1.pt", map_location="cpu")
    # x = torch.cat((cat_features, prop_features, tweet_features, des_features), dim=1)
    x = torch.load(args.path + "features.pt", map_location="cpu")
    print('x.shape:',x.shape)
    
    print("loading edges & label...")
    edge_index = torch.load(args.path + "edge_index.pt", map_location="cpu")
    edge_type = torch.load(args.path + "edge_type.pt", map_location="cpu").unsqueeze(-1)
    # edge_weight = torch.load(args.path + "edge_weight.pt", map_location="cpu")
    label = torch.load(args.path + "labels_bot.pt", map_location="cpu")
    data = Data(x=x, edge_index = edge_index, edge_attr=edge_type, y=label)
    node_num = x.shape[0]
    sample_idx = shuffle(np.array(range(node_num)),
                                 random_state=args.seed)
    data.train_idx = sample_idx[:int(0.7 * node_num)]
    data.valid_idx = sample_idx[int(0.7 * node_num):int(0.9 * node_num)]
    data.test_idx = sample_idx[int(0.9 * node_num):]
    
    # print("loading index...")
    # data.train_idx = torch.load(args.path + "train_idx.pt", map_location="cpu")
    # data.valid_idx = torch.load(args.path + "val_idx.pt", map_location="cpu")
    # data.test_idx = torch.load(args.path + "test_idx.pt", map_location="cpu")
    
    return data
    
class SHGNDetector(pl.LightningModule):
    def __init__(self, args):
        super(SHGNDetector, self).__init__()
    
        self.lr = args.lr
        self.l2_reg = args.l2_reg
    
        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(788, args.linear_channels)

        self.HGN_layer1 = SimpleHGN(num_edge_type=7, in_channels=args.linear_channels, out_channels=args.out_channel, rel_dim=args.rel_dim, beta=args.beta)
        self.HGN_layer2 = SimpleHGN(num_edge_type=7, in_channels=args.linear_channels, out_channels=args.out_channel, rel_dim=args.rel_dim, beta=args.beta, final_layer=True)

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
        # cat_features = train_batch.x[:, :args.cat_num]
        # prop_features = train_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
        # tweet_features = train_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
        # des_features = train_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
        
        user_features = train_batch.x
        label = train_batch.y
        
        edge_index = train_batch.edge_index
        edge_type = train_batch.edge_attr.view(-1)
        
        # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features)
        loss = self.CELoss(pred, label)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            # cat_features = val_batch.x[:, :args.cat_num]
            # prop_features = val_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = val_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = val_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            user_features = val_batch.x
            label = val_batch.y
        
            edge_index = val_batch.edge_index
            edge_type = val_batch.edge_attr.view(-1)
            
            # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            pred_binary = torch.argmax(pred, dim=1)
            
            # print(self.label[val_batch].size())

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            
            self.log("val_acc", acc, prog_bar=True, on_step=False, batch_size=label.size(0))
            self.log("val_f1", f1, prog_bar=True, on_step=False, batch_size=label.size(0))

            # print("acc: {} f1: {}".format(acc, f1))
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            # cat_features = test_batch.x[:, :args.cat_num]
            # prop_features = test_batch.x[:, args.cat_num: args.cat_num + args.numeric_num]
            # tweet_features = test_batch.x[:, args.cat_num+args.numeric_num: args.cat_num+args.numeric_num+args.tweet_channel]
            # des_features = test_batch.x[:, args.cat_num+args.numeric_num+args.tweet_channel: args.cat_num+args.numeric_num+args.tweet_channel+args.des_channel]
            
            user_features = test_batch.x
            label = test_batch.y[:args.test_batch_size]
            edge_index = test_batch.edge_index
            edge_type = test_batch.edge_attr.view(-1)
            
            # user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            # user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            # user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            # user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            # user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
            user_features, _ = self.HGN_layer2(user_features, edge_index, edge_type, alpha)

            user_features = self.drop(self.ReLU(self.out1(user_features)))
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
            results_df.to_csv('SimpleHGN_mgtab.csv', mode='a', header=not os.path.exists(f'SimpleHGN_mgtab.csv'), index=False)



            
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }


parser = argparse.ArgumentParser(description="SimpleHGN")

parser.add_argument("--path", type=str, default="/dev/shm/mgtab/", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--rel_dim", type=int, default=200, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--batch_size", type=int, default=512, help="description channel")
parser.add_argument("--epochs", type=int, default=200, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=0, help="description channel")
parser.add_argument("--seed", type=int, default=None, help="random")
parser.add_argument("--beta", type=float, default=0.05, help="description channel")
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

    # train_dataset = BotDataset(name="train")
    # valid_dataset = BotDataset(name="valid")
    # test_dataset = BotDataset(name="test")
    data = load_data(args)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    train_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.train_idx, batch_size=args.batch_size, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.valid_idx, batch_size=args.batch_size)
    test_loader = NeighborLoader(data, num_neighbors=[256]*4, input_nodes=data.test_idx, batch_size=args.test_batch_size)
    
    model = SHGNDetector(args)
    trainer = pl.Trainer(num_nodes=1, max_epochs=args.epochs, precision=16, log_every_n_steps=1, callbacks=[checkpoint_callback,early_stop_callback])
    
    # trainer.fit(model, train_loader, valid_loader, ckpt_path='./lightning_logs/version_0/checkpoints/val_acc=0.9033.ckpt')

    # dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    # best_path = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_path = './lightning_logs/version_0/checkpoints/val_acc=0.9033.ckpt'
    best_model = SHGNDetector.load_from_checkpoint(checkpoint_path=best_path, args=args)
    trainer.test(best_model, test_loader, verbose=True)
    
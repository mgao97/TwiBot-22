from SEPN.sep_u import SEP_U
from SEPG.sep_g import SEP_G
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from torch_geometric.data import Data
from BackBone.rgcn import FACNConv as RGCNConv
from BackBone.self_attention import SelfAttention
import argparse
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,auc,roc_auc_score,precision_recall_curve,auc

import pandas as pd
import os
import numpy as np
import random

PWD = '/dev/shm/twi20/'


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def edge_mask(edge_index, edge_attr, pe):
    # each edge has a probability of pe to be removed
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr


def feature_mask(x, drop_prob):
    # each feature channel has a probability of being masked
    drop_mask = torch.empty(
        (x.size(1), ), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(x.device)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def relational_undirected(edge_index, edge_type):
    device = edge_index.device
    relation_num = edge_type.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_type.clone()
    r_edge = []
    for i in range(relation_num):
        e1 = edge_index[:, edge_type == i].unique(dim=1)
        e2 = e1.flip(0)
        edges = torch.cat((e1, e2), dim=1)
        r_edge.append(edges)
    edge_type = torch.cat(
        [torch.tensor([i] * e.shape[1]) for i, e in enumerate(r_edge)],
        dim=0).to(device)
    edge_index = torch.cat(r_edge, dim=1)

    return edge_index, edge_type


class BotRGCN(nn.Module):

    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        self.node_num = args.node_num
        self.pe = args.pe
        self.pf = args.pf
        input_dimension = args.input_dim
        embedding_dimension = args.hidden_dim

        self.linear_relu_des = nn.Sequential(
            nn.Linear(self.des_size, int(input_dimension / 4)), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(self.tweet_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(self.cat_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(
            nn.Linear(input_dimension, embedding_dimension),
            nn.PReLU(embedding_dimension))

        self.rgcn1 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)
        self.rgcn2 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension))

        self.relu = nn.LeakyReLU()

    def forward(self, data, return_attention=False):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        if self.training:
            edge_index, edge_type = edge_mask(edge_index, edge_type, self.pe)

        num_prop = x[:, :self.num_prop_size]
        tweet = x[:, self.num_prop_size:self.num_prop_size + self.tweet_size]
        cat_prop = x[:,
                     self.num_prop_size + self.tweet_size:self.num_prop_size +
                     self.tweet_size + self.cat_prop_size]
        des = x[:, self.num_prop_size + self.tweet_size + self.cat_prop_size:]
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn1(x, edge_index, edge_type, return_attention)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


class SEBot(nn.Module):

    def __init__(self, args):
        super(SEBot, self).__init__()
        self.args = args
        self.sep_u = SEP_U(self.args)
        self.sep_g = SEP_G(self.args)
        self.backbone = BotRGCN(self.args)
        # self.feature_encoder = FeatureEncoder(self.args)

        self.classifier = nn.Sequential(
            nn.Linear(self.args.hidden_dim * 3, self.args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.hidden_dim, self.args.num_classes))
        self.proj_u = nn.Sequential(
            nn.Linear(self.args.hidden_dim,
                      self.args.proj_dim), nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim))
        self.proj_g = nn.Sequential(
            nn.Linear(self.args.hidden_dim,
                      self.args.proj_dim), nn.LeakyReLU(),
            nn.Linear(self.args.proj_dim, self.args.hidden_dim))
        self.self_attention = SelfAttention(self.args.hidden_dim)
        self.test_results = []

    # https://blog.csdn.net/weixin_44966641/article/details/120382198
    # def infonce_loss(self,
    #                  emb_i,
    #                  emb_j,
    #                  temperature=0.1):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
    #     batch_size = emb_i.shape[0]
    #     negatives_mask = (
    #         ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(
    #             self.args.device).float()  # (2*bs, 2*bs)
    #     z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
    #     z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

    #     representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
    #     similarity_matrix = torch.mm(representations, representations.t())

    #     sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
    #     sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
    #     positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

    #     nominator = torch.exp(positives / temperature)  # 2*bs
    #     denominator = negatives_mask * torch.exp(
    #         similarity_matrix / temperature)  # 2*bs, 2*bs

    #     loss_partial = -torch.log(
    #         nominator / torch.sum(denominator, dim=1))  # 2*bs
    #     loss = torch.sum(loss_partial) / (2 * batch_size)
    #     return loss

    def infonce_loss_efficient(self, z_i, z_j, temperature=0.5, chunk_size=1024):
        """
        内存高效的InfoNCE损失函数实现
        通过分块计算相似度矩阵来减少内存使用
        """
        batch_size = z_i.size(0)
        
        # 将表示向量归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 将所有表示向量拼接在一起
        representations = torch.cat([z_i, z_j], dim=0)
        
        # 初始化损失
        total_loss = 0.0
        n_chunks = (batch_size * 2 + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            # 计算当前块的起始和结束索引
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size * 2)
            current_chunk = representations[start_idx:end_idx]
            
            # 计算当前块与所有表示向量的相似度
            similarity_matrix_chunk = torch.mm(current_chunk, representations.t()) / temperature
            
            # 创建当前块的标签矩阵
            labels_chunk = torch.zeros(end_idx - start_idx, batch_size * 2, device=z_i.device)
            
            # 设置正样本对的位置
            for k in range(start_idx, end_idx):
                pos_idx = (k + batch_size) % (2 * batch_size)
                labels_chunk[k - start_idx, pos_idx] = 1.0
            
            # 计算当前块的损失
            logits_max, _ = torch.max(similarity_matrix_chunk, dim=1, keepdim=True)
            logits = similarity_matrix_chunk - logits_max.detach()  # 数值稳定性
            
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            
            # 计算正样本对的均值
            mean_log_prob_pos = (labels_chunk * log_prob).sum(1) / labels_chunk.sum(1)
            
            # 累加损失
            loss = -mean_log_prob_pos.mean()
            total_loss += loss * (end_idx - start_idx) / (batch_size * 2)
        
        return total_loss

    def forward(self, batch):
        out_u = self.sep_u(batch)
        out_g = self.sep_g(batch['subgraphs'],
                           batch['data'].x)  # do not contain support nodes
        out_c = self.backbone(batch['data'])
        out_u = out_u[:self.args.node_num, :]
        out_c = out_c[:self.args.node_num, :]

        loss1 = self.infonce_loss(self.proj_u(out_u), self.proj_u(out_c),
                                  self.args.temperature)
        loss2 = self.infonce_loss(self.proj_g(out_g), self.proj_g(out_c),
                                  self.args.temperature)
        if self.training:
            # Training
            train_out = torch.cat([out_u, out_g, out_c],
                                  dim=1)[batch['data'].train_idx]
            train_out = self.classifier(train_out)
            loss = F.cross_entropy(train_out,
                                   batch['data'].y[batch['data'].train_idx])
            loss = loss + loss1 * self.args.alpha1 + loss2 * self.args.alpha2
            return loss
        else:
            # Validation
            val_out = torch.cat([out_u, out_g, out_c],
                                dim=1)[batch['data'].val_idx]
            val_out = self.classifier(val_out)
            val_loss = F.cross_entropy(val_out,
                                       batch['data'].y[batch['data'].val_idx])
            val_acc = accuracy_score(
                batch['data'].y[batch['data'].val_idx].cpu().numpy(),
                torch.argmax(val_out, dim=1).cpu().numpy())

            # Test
            test_out = torch.cat([out_u, out_g, out_c],
                                 dim=1)[batch['data'].test_idx]
            
            torch.save(test_out, '/dev/shm/twi20/res/sebot_twi20_emb.pt')
            torch.save(batch['data'].y[batch['data'].test_idx].cpu(), 'sebot_twi20_y.pt')

            test_out = self.classifier(test_out)
            test_loss = F.cross_entropy(
                test_out, batch['data'].y[batch['data'].test_idx])
            test_label = batch['data'].y[batch['data'].test_idx].cpu().numpy()
            test_pred = torch.argmax(test_out, dim=1).cpu().numpy()

            test_acc = accuracy_score(test_label, test_pred)
            test_f1 = f1_score(test_label, test_pred)
            test_recall = recall_score(test_label, test_pred)
            test_precision = precision_score(test_label, test_pred)
            test_precision1, test_recall1, _ = precision_recall_curve(test_label, test_pred)
            test_aucpr = auc(test_recall1, test_precision1)

            # 计算精确率为80%时的召回率
            recall_at_p80 = 0
            for pi, ri in zip(test_precision1, test_recall1):
                if pi >= 0.8:
                    recall_at_p80 = ri
                    break

            # 计算精确率为85%时的召回率
            recall_at_p85 = 0
            for pi, ri in zip(test_precision1, test_recall1):
                if pi >= 0.85:
                    recall_at_p85 = ri
                    break

            # 计算精确率为90%时的召回率
            recall_at_p90 = 0
            for pi, ri in zip(test_precision1, test_recall1):
                if pi >= 0.9:
                    recall_at_p90 = ri
                    break


            test_rocauc = roc_auc_score(test_label, test_pred)
            # if test_acc > 0.869:
            #     print('large test_acc: ', test_acc)
            #     _ = self.backbone(batch['data'], return_attention=True)

            # 打印这些指标
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test AUC-PR: {test_aucpr:.4f}")
            print(f"Test ROC-AUC: {test_rocauc:.4f}")
            results = []
            # 记录结果
            results.append({
                'seed': args.seed,
                'Accuracy': test_acc,
                'Precision': test_precision,
                'Recall': test_recall,
                'F1': test_f1,
                'AUCROC': test_rocauc,
                'AUC-PR': test_aucpr,
                'Recall@P80': recall_at_p80,
                'Recall@P85': recall_at_p85,
                'Recall@P90': recall_at_p90
            })
            # 生成表格
            results_df = pd.DataFrame(results)
            print(results_df.to_markdown(index=False))

            # 保存结果到CSV
            results_df.to_csv('SEBot_twi20.csv', mode='a', header=not os.path.exists(f'SEBot_twi20.csv'), index=False)

            self.test_results.append(
                [test_acc, test_f1, test_recall, test_precision, test_aucpr, test_rocauc])
            return val_acc, val_loss.item(), test_acc, test_loss.item()

    def get_test_results(self):
        return self.test_results


class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()
        # Random Seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.args = args

        self.load_data()  # make sure load data before init SEBot
        self.sebot = SEBot(self.args).to(self.args.device)
        self.save_top_k = args.save_top_k
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.test_results = []

    def load_data(self):
        t_path = os.path.join(
            PWD, 'trees',
            '%s_%s.pickle' % (self.args.dataset, self.args.tree_depth))
        with open(t_path, 'rb') as fp:
            self.layer_data = pickle.load(fp)
        g_path = os.path.join(
            PWD, 'subgraphs',
            '%s_%s.pickle' % (self.args.dataset, self.args.tree_depth))
        with open(g_path, 'rb') as fp:
            self.subgraphs = pickle.load(fp)
        # self.args.num_features = self.subgraphs[0]['node_features'].size(1)
        self.args.num_features = self.args.hidden_dim

        # path = './dataset/' + self.args.dataset + '/'
        path = '/dev/shm/twi20/processed_data/'
        edge_index = torch.load(path + 'edge_index.pt')
        edge_type = torch.load(path + 'edge_type.pt')
        edge_index, edge_type = relational_undirected(edge_index, edge_type)
        self.args.num_relations = edge_type.max() + 1

        x = torch.cat([
            torch.load(path + 'num_properties_tensor.pt'),
            torch.load(path + 'tweets_tensor.pt'),
            torch.load(path + 'cat_properties_tensor.pt'),
            torch.load(path + 'des_tensor.pt')
        ],
                      dim=1)
        sample_idx = list(range(self.args.node_num))

        label = torch.load(path + 'label.pt')
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                    y=label).to(self.args.device)
        data.train_idx = sample_idx[:int(0.7 * args.node_num)]
        data.val_idx = sample_idx[int(0.7 * args.node_num):int(0.9 *
                                                               args.node_num)]
        data.test_idx = sample_idx[int(0.9 * args.node_num):]

        # data.edge_mask = TTA(data)

        self.data = data

    def organize_val_log(self, val_loss, val_acc, epoch):
        if val_loss < self.best_loss:
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

    def train(self):
        self.optimizer = torch.optim.AdamW(self.sebot.parameters(),
                                           lr=self.args.lr,
                                           weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=16, eta_min=0)

        val_accs = []
        val_losses = []
        test_accs = []

        for epoch in range(self.args.epochs):

            self.sebot.train()

            batch = {
                'data': self.data.to(self.args.device),
                'layer_data': self.layer_data,  # total graph tree
                'subgraphs': self.subgraphs  # subgraph trees
            }
            self.optimizer.zero_grad()
            loss = self.sebot(batch)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Validation
            val_acc, val_loss, test_acc, _ = self.eval(batch)
            # self.scheduler.step(val_loss)
            print('epoch: %d, val_acc: %.4f, val_loss: %.4f, test_acc: %.4f' %
                  (epoch, val_acc, val_loss, test_acc))
            self.organize_val_log(val_loss, val_acc, epoch)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            
            if self.patience > self.args.patience:
                break

        test_results = self.sebot.get_test_results()
        test_results = np.array(test_results)
        # 选择验证集上loss最小的self.save_top_k个结果打印
        val_losses = np.array(val_losses)
        min_loss_index = val_losses.argsort()[:self.save_top_k]
        for i in min_loss_index:
            print(
                'epoch: %d, test_acc: %.4f, test_f1: %.4f, test_recall: %.4f, test_precision: %.4f, test_aucpr: %.4f, test_rocauc: %.4f'
                % (i, test_results[i][0], test_results[i][1],
                   test_results[i][2], test_results[i][3], test_results[i][4], test_results[i][5]))

        return test_accs[val_losses.argmin()]

    def eval(self, batch):
        self.sebot.eval()
        return self.sebot(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEP')
    parser.add_argument('--dataset', type=str, default='twibot-20')
    parser.add_argument('--node_num', type=int, default=11826)
    parser.add_argument('--tree_depth', type=int, default=4)

    parser.add_argument('--cat_num', type=int, default=3)
    parser.add_argument('--prop_num', type=int, default=5)
    parser.add_argument('--des_num', type=int, default=768)
    parser.add_argument('--tweet_num', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--conv', type=str, default='GCN')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--proj_dim', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--alpha1', type=float,
                        default=0.1)  # node self-supervised loss
    parser.add_argument('--alpha2', type=float,
                        default=0.1)  # subgraph self-supervised loss
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_convs', type=int, default=3)
    parser.add_argument('--link_input', action='store_true', default=False)
    parser.add_argument('-gp','--global-pooling',type=str,default="average",choices=["sum", "average"],help='Pooling for over nodes: sum or average')
    parser.add_argument('--pe', type=float, default=0.2)  # edge dropout rate
    parser.add_argument('--pf', type=float, default=0.2)  # edge dropout ratee

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument('--conv_dropout', type=float, default=0.5)
    parser.add_argument('--pooling_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_top_k', type=int,default=6)  # save top k models with best validation loss

    args = parser.parse_args()

    global results
    results = []

    print(args.link_input)
    # args.device = torch.device(
    #     "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cpu")
    trainer = Trainer(args)
    results = []
    test_acc = trainer.train()
    # print('test_acc: ', test_acc)

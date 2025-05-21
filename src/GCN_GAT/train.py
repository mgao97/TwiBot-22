import os
from argparse import ArgumentParser
from utils import null_metrics, calc_metrics, is_better
import torch
from dataset import get_train_data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import torch.nn as nn
from model import BotGAT, BotGCN, BotRGCN
import os
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-20')
parser.add_argument('--mode', type=str, default='GAT')
parser.add_argument('--visible', type=bool, default=False)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_up', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=97)
parser.add_argument('--save_embedding', type=bool, default=True)  # 添加保存embedding的参数
args = parser.parse_args()

dataset_name = args.dataset
mode = args.mode
visible = args.visible
save_embedding = args.save_embedding  # 获取是否保存embedding的参数

assert mode in ['GCN', 'GAT', 'RGCN']
assert dataset_name in ['cresci-2015', 'Twibot-20', 'Twibot-22']

data = get_train_data(dataset_name)

hidden_dim = args.hidden_dim
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay
max_epoch = args.max_epoch
batch_size = args.batch_size
no_up = args.no_up


def forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader):
    model.train()
    all_label = []
    all_pred = []
    
    ave_loss = 0.0
    cnt = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        n_batch = batch.batch_size

        
        out = model(batch.des_embedding,
                    batch.tweet_embedding,
                    batch.num_property_embedding,
                    batch.cat_property_embedding,
                    batch.edge_index,
                    batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
        loss.backward()
        optimizer.step()
    ave_loss /= cnt
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
    if visible:
        print(plog)
    

    val_metrics = validation(epoch, 'validation', model, loss_fn, val_loader)
    return val_metrics


@torch.no_grad()
def validation(epoch, name, model, loss_fn, loader, save_emb=False):
    model.eval()
    all_label = []
    all_pred = []
    all_embedding = []  # 用于存储所有样本的embedding
    all_indices = []    # 用于存储所有样本的索引
    ave_loss = 0.0
    cnt = 0.0
    for batch in loader:
        batch = batch.to(device)
        n_batch = batch.batch_size
        # 获取embedding和预测结果
        if save_emb:
            out, embedding = model(batch.des_embedding,
                        batch.tweet_embedding,
                        batch.num_property_embedding,
                        batch.cat_property_embedding,
                        batch.edge_index,
                        batch.edge_type,
                        return_embedding=True)  # 添加参数以返回embedding
            # 保存当前批次的embedding和对应的索引
            all_embedding.append(embedding[:n_batch].cpu())
            all_indices.extend(batch.n_id[:n_batch].cpu().numpy().tolist())
        else:
            out = model(batch.des_embedding,
                        batch.tweet_embedding,
                        batch.num_property_embedding,
                        batch.cat_property_embedding,
                        batch.edge_index,
                        batch.edge_type)
        label = batch.y[:n_batch]
        out = out[:n_batch]
        all_label += label.data
        all_pred += out
        loss = loss_fn(out, label)
        ave_loss += loss.item() * n_batch
        cnt += n_batch
    ave_loss /= cnt
    all_label = torch.stack(all_label)
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
    if visible:
        print(plog)
    # 如果需要保存embedding
    if save_emb and all_embedding:
        # 合并所有批次的embedding
        all_embedding = torch.cat(all_embedding, dim=0)
        # 创建embedding目录
        emb_dir = 'embeddings'
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        # 保存embedding和对应的索引
        emb_path = os.path.join(emb_dir, '{}_{}_{}_embedding.pt'.format(dataset_name, mode, name))
        torch.save({
            'embedding': all_embedding,
            'indices': all_indices
        }, emb_path)
        print(f'已保存{name}集embedding到: {emb_path}')
    return metrics


def train():
    print(data)
    train_loader = NeighborLoader(data,
                                  num_neighbors=[64] * 4,
                                  batch_size=batch_size,
                                  input_nodes=data.train_idx,
                                  shuffle=True)
    val_loader = NeighborLoader(data,
                                num_neighbors=[64] * 4,
                                batch_size=batch_size,
                                input_nodes=data.val_idx)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[64] * 4,
                                 batch_size=batch_size,
                                 input_nodes=data.test_idx)
    if mode == 'GAT':
        model = BotGAT(hidden_dim=hidden_dim,
                       dropout=dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'GCN':
        model = BotGCN(hidden_dim=hidden_dim,
                       dropout=dropout,
                       num_prop_size=data.num_property_embedding.shape[-1],
                       cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
    elif mode == 'RGCN':
        model = BotRGCN(hidden_dim=hidden_dim,
                        dropout=dropout,
                        num_prop_size=data.num_property_embedding.shape[-1],
                        cat_prop_size=data.cat_property_embedding.shape[-1],
                        num_relations=data.edge_type.max().item() + 1).to(device)
    else:
        raise KeyError
    best_val_metrics = null_metrics()
    best_state_dict = None
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(max_epoch), ncols=0)
    cnt = 0
    for epoch in pbar:
        val_metrics = forward_one_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader)
        if is_better(val_metrics, best_val_metrics):
            best_val_metrics = val_metrics
            best_state_dict = model.state_dict()
            cnt = 0
            # 保存当前最佳模型
            checkpoint_dir = 'checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_path = os.path.join(checkpoint_dir, '{}_{}_epoch{}_acc{:.4f}.pt'.format(
                dataset_name, mode, epoch, val_metrics['acc']))
            torch.save(best_state_dict, model_path)
            print(f'保存最佳模型到: {model_path}')
        else:
            cnt += 1
        pbar.set_postfix_str('val acc {} no up cnt {}'.format(val_metrics['acc'], cnt))
        if cnt == no_up:
            break
    model.load_state_dict(best_state_dict)
    results = []
    test_metrics = validation(max_epoch, 'test', model, loss_fn, test_loader, save_emb=save_embedding)
    test_metrics['seed'] = args.seed
    results.append(test_metrics)
    # 生成表格
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))

    # 保存结果到CSV
    results_df.to_csv(f'GAT_{args.dataset}.csv', mode='a', header=not os.path.exists(f'GAT_{args.dataset}.csv'), index=False)
    
    # 保存最终的最佳模型
    final_model_path = os.path.join('checkpoints', '{}_{}_{:.4f}.pt'.format(dataset_name, mode, test_metrics['acc']))
    torch.save(best_state_dict, final_model_path)
    print(f'保存最终模型到: {final_model_path}')
    # torch.save(best_state_dict, 'checkpoints/{}_{}.pt'.format(dataset_name, test_metrics['acc']))
    for key, value in test_metrics.items():
        print(key, value)


if __name__ == '__main__':
    train()
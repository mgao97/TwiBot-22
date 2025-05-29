#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gsp


class FeatureEncoder(nn.Module):

    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        self.args = args
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
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

    def forward(self, x):
        if self.args.dataset == 'twibot-20':
            num_prop = x[:, :self.num_prop_size]
            tweet = x[:,
                      self.num_prop_size:self.num_prop_size + self.tweet_size]
            cat_prop = x[:, self.num_prop_size +
                         self.tweet_size:self.num_prop_size + self.tweet_size +
                         self.cat_prop_size]
            des = x[:,
                    self.num_prop_size + self.tweet_size + self.cat_prop_size:]
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        return x


class SEPooling(MessagePassing):

    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return out
        # return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SEP_G(torch.nn.Module):

    def __init__(self, args):
        super(SEP_G, self).__init__()
        self.args = args
        self.num_features = args.input_dim  # input_dim
        self.nhid = args.hidden_dim  # hidden dim
        self.num_classes = args.num_classes  # output dim
        self.num_features = args.num_features
        self.dropout_ratio = args.pooling_dropout
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()
        self.global_pool = gsp if args.global_pooling == 'sum' else gap
        self.classifier = self.get_classifier()

        self.feature_encoder = FeatureEncoder(args)
    
    def __process_layer_edgeIndex(self, batch_data, layer=0):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            # Check if 'node_size' key exists
            if 'node_size' not in graph:
                # If 'node_size' doesn't exist, create it with default values
                if 'G' in graph:
                    # If graph object exists, use its node count
                    node_count = len(graph['G'].nodes)
                    graph['node_size'] = {layer: node_count}
                else:
                    # Fallback to a default size
                    graph['node_size'] = {layer: 1}
            
            # Ensure the layer key exists in node_size
            if layer not in graph['node_size']:
                if 'G' in graph:
                    graph['node_size'][layer] = len(graph['G'].nodes)
                else:
                    graph['node_size'][layer] = 1
                    
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            
            # Check if 'graph_mats' exists
            if 'graph_mats' not in graph or layer not in graph['graph_mats']:
                # Create empty edge matrix if missing
                if 'graph_mats' not in graph:
                    graph['graph_mats'] = {}
                graph['graph_mats'][layer] = torch.zeros((2, 0), dtype=torch.long)
                
            edge_mat_list.append(graph['graph_mats'][layer] + start_idx[i])
        
        # Handle case where edge_mat_list is empty
        if not edge_mat_list:
            return torch.zeros((2, 0), dtype=torch.long).to(self.args.device)
            
        edge_index = torch.cat(edge_mat_list, 1)
        return edge_index.to(self.args.device)

    # def __process_layer_edgeIndex(self, batch_data, layer=0):  # 生成对应层的边索引，因为
    #     edge_mat_list = []
    #     start_idx = [0]
    #     for i, graph in enumerate(batch_data):
    #         # print('start_idx[i]:',i, start_idx[i])
    #         # print('graph:',graph)
    #         start_idx.append(start_idx[i] + graph['node_size'][layer])
    #         edge_mat_list.append(graph['graph_mats'][layer] + start_idx[i])
    #     edge_index = torch.cat(edge_mat_list, 1)
    #     return edge_index.to(self.args.device)

    # def __process_sep_edgeIndex(self, batch_data, layer=1):
    #     edge_mat_list = []
    #     start_pdx = [0]
    #     start_idx = [0]
    #     for i, graph in enumerate(batch_data):
    #         start_pdx.append(start_pdx[i] + graph['node_size'][layer - 1])
    #         start_idx.append(start_idx[i] + graph['node_size'][layer])
    #         edge_mat_list.append(
    #             torch.LongTensor(graph['edges'][layer]) +
    #             torch.LongTensor([start_idx[i], start_pdx[i]]))
    #     edge_index = torch.cat(edge_mat_list, 0).T
    #     return edge_index.to(self.args.device)

    # def __process_sep_edgeIndex(self, batch_data, layer=1):
    #     edge_mat_list = []
    #     start_pdx = [0]
    #     start_idx = [0]
    #     for i, graph in enumerate(batch_data):
    #         # 检查 'node_size' 键是否存在
    #         if 'node_size' not in graph:
    #             # 如果不存在，创建默认的 node_size 字典
    #             if 'G' in graph:
    #                 # 如果有图对象，使用图中节点数量
    #                 graph['node_size'] = {
    #                     layer-1: len(graph['G'].nodes),
    #                     layer: len(graph['G'].nodes)
    #                 }
    #             else:
    #                 # 否则使用默认值
    #                 graph['node_size'] = {layer-1: 1, layer: 1}
            
    #         # 确保 layer 和 layer-1 键存在于 node_size 字典中
    #         if layer-1 not in graph['node_size']:
    #             graph['node_size'][layer-1] = 1
    #         if layer not in graph['node_size']:
    #             graph['node_size'][layer] = 1
                
    #         start_pdx.append(start_pdx[i] + graph['node_size'][layer - 1])
    #         start_idx.append(start_idx[i] + graph['node_size'][layer])
            
    #         # 检查 'edges' 键是否存在
    #         if 'edges' not in graph or layer not in graph['edges']:
    #             # 如果不存在，创建空边列表
    #             if 'edges' not in graph:
    #                 graph['edges'] = {}
    #             graph['edges'][layer] = []
                
    #         edge_mat_list.append(
    #             torch.LongTensor(graph['edges'][layer]) +
    #             torch.LongTensor([start_idx[i], start_pdx[i]]))
        
    #     # 如果没有边，返回空张量
    #     if not edge_mat_list:
    #         return torch.zeros((2, 0), dtype=torch.long).to(self.args.device)
            
    #     edge_index = torch.cat(edge_mat_list, 0).T
    #     return edge_index.to(self.args.device)
    def __process_sep_edgeIndex(self, batch_data, layer=1):
        edge_mat_list = []
        start_pdx = [0]
        start_idx = [0]
        for i, graph in enumerate(batch_data):
            # Check if 'node_size' key exists
            if 'node_size' not in graph:
                # If 'node_size' doesn't exist, create it with default values
                if 'G' in graph:
                    # If graph object exists, use its node count
                    node_count = len(graph['G'].nodes)
                    graph['node_size'] = {layer-1: node_count, layer: node_count}
                else:
                    # Fallback to a default size
                    graph['node_size'] = {layer-1: 1, layer: 1}
            
            # Ensure the layer keys exist in node_size
            if layer-1 not in graph['node_size']:
                graph['node_size'][layer-1] = 1
            if layer not in graph['node_size']:
                graph['node_size'][layer] = 1
                
            start_pdx.append(start_pdx[i] + graph['node_size'][layer - 1])
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            
            # Check if 'edges' key exists
            if 'edges' not in graph or layer not in graph['edges']:
                # If 'edges' doesn't exist, create an empty list
                if 'edges' not in graph:
                    graph['edges'] = {}
                graph['edges'][layer] = []
            
            # Handle empty edge lists
            if len(graph['edges'][layer]) == 0:
                # Skip this graph if it has no edges
                continue
                
            edge_mat_list.append(
                torch.LongTensor(graph['edges'][layer]) +
                torch.LongTensor([start_idx[i], start_pdx[i]]))
        
        # If no valid edges were found, return an empty tensor
        if not edge_mat_list:
            return torch.zeros((2, 0), dtype=torch.long).to(self.args.device)
            
        edge_index = torch.cat(edge_mat_list, 0).T
        return edge_index.to(self.args.device)

    

    def __process_sep_size(self, batch_data, layer=1):
        size = [(graph['node_size'][layer], graph['node_size'][layer - 1])
                for graph in batch_data]
        return np.array(size).sum(
            axis=0)  # size = [sum(node_size[layer-1]), sum(node_size[layer])]

    def __process_batch(self, batch_data, layer=0):
        batch = [[i] * graph['node_size'][layer]
                 for i, graph in enumerate(batch_data)]
        batch = reduce(lambda x, y: x + y, batch)
        return torch.tensor(batch, dtype=torch.long).to(self.args.device)

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.num_convs):
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT':
                conv = GATConv(_input_dim,
                               _output_dim,
                               self.args.num_head,
                               concat=False)
            elif self.args.conv == 'Cheb':
                conv = ChebConv(_input_dim, _output_dim, K=2)
            elif self.args.conv == 'SAGE':
                conv = SAGEConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT2':
                conv = GATv2Conv(_input_dim,
                                 _output_dim,
                                 self.args.num_head,
                                 concat=False)
            elif self.args.conv == 'Transformer':
                conv = TransformerConv(_input_dim,
                                       _output_dim,
                                       2,
                                       concat=False)
            elif self.args.conv == 'GIN':
                conv = GINConv(nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ),
                               train_eps=False)
            convs.append(conv)
            _input_dim = _output_dim
        return convs

    def get_sepool(self):
        pools = nn.ModuleList()
        _input_dim = self.nhid
        _output_dim = self.nhid
        for _ in range(self.args.tree_depth - 1):
            pool = SEPooling(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ))
            pools.append(pool)
            _input_dim = _output_dim
        return pools

    def get_classifier(self):
        init_dim = self.nhid * self.args.num_convs
        if self.args.link_input:
            init_dim += self.num_features
        return nn.Sequential(nn.Linear(init_dim, self.nhid), nn.ReLU(),
                             nn.Dropout(p=self.dropout_ratio),
                             nn.Linear(self.nhid, self.nhid))

    def forward(self, batch_data, x):
        # batch_data: graph data list
        # x = self.feature_encoder(x)
        # x = xIn = torch.cat([t['node_features'] for t in batch_data],
        #                     dim=0).to(self.args.device)
        x = self.feature_encoder(x)

        x = xIn = torch.cat([x[graph['nodelist']] for graph in batch_data],
                            dim=0).to(self.args.device)

        xs = []
        for _ in range(self.args.num_convs):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch_data, _)
            x = F.relu(self.convs[_](x, edge_index))
            # sep
            if _ < self.args.tree_depth - 1:
                edge_index = self.__process_sep_edgeIndex(batch_data, _ + 1)
                size = self.__process_sep_size(batch_data, _ + 1)
                x = F.relu(self.sepools[_](x, edge_index, size=size))
            xs.append(x)

        pooled_xs = []
        if self.args.link_input:
            batch = self.__process_batch(batch_data, 0)
            pooled_x = self.global_pool(xIn, batch)
            pooled_xs.append(pooled_x)
        for _, x in enumerate(xs):
            batch = self.__process_batch(batch_data,
                                         min(_ + 1, self.args.tree_depth - 1))
            pooled_x = self.global_pool(x, batch)
            pooled_xs.append(pooled_x)

        # For jumping knowledge scheme
        x = torch.cat(pooled_xs, dim=1)
        # For Classification
        x = self.classifier(x)
        return x

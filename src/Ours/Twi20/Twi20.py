import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# 自定义 Dataset 类
class Twi20Dataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 通过索引获取节点的特征和标签
        node_idx = self.indices[idx]
        x = self.data.x[node_idx]      # 特征
        y = self.data.label[node_idx]  # 标签
        edge_index = self.data.edge_index
        edge_type = self.data.edge_type
        
        return x,y

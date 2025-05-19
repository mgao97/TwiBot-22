import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from argparse import ArgumentParser
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)

# 数据路径
path = '/dev/shm/mgtab/'

print('加载数据...')
# 加载图结构和节点特征
edge_index = torch.load(path + 'edge_index.pt')
features = torch.load(path + 'features.pt').numpy()
labels = torch.load(path + 'labels_bot.pt').numpy()

# 可选：加载边类型和边权重
try:
    edge_type = torch.load(path + 'edge_type.pt')
    print(f'边类型加载成功，形状: {edge_type.shape}')
except FileNotFoundError:
    print('边类型文件不存在')
    edge_type = None

try:
    edge_weight = torch.load(path + 'edge_weight.pt')
    print(f'边权重加载成功，形状: {edge_weight.shape}')
except FileNotFoundError:
    print('边权重文件不存在')
    edge_weight = None

print(f'特征形状: {features.shape}')
print(f'标签形状: {labels.shape}')
print(f'边索引形状: {edge_index.shape}')

# 划分训练集、验证集和测试集 (7:2:1)
num_nodes = features.shape[0]
indices = np.random.permutation(num_nodes)
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
train_size = int(num_nodes * train_ratio)
val_size = int(num_nodes * val_ratio)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

print('数据加载完成')
print(f'训练集大小: {len(train_idx)}')
print(f'验证集大小: {len(val_idx)}')
print(f'测试集大小: {len(test_idx)}')

if __name__ == '__main__':
    train_x = features[train_idx]
    train_y = labels[train_idx]
    val_x = features[val_idx]
    val_y = labels[val_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]
    
    print('训练中...')
    cls = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    cls.fit(train_x, train_y)
    print('训练完成')

    val_pred = cls.predict(val_x)
    test_pred = cls.predict(test_x)

    val_acc = accuracy_score(val_y, val_pred)
    val_f1 = f1_score(val_y, val_pred)
    val_recall = recall_score(val_y, val_pred)
    val_precision = precision_score(val_y, val_pred)

    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred)
    test_recall = recall_score(test_y, test_pred)
    test_precision = precision_score(test_y, test_pred)

    print('验证集结果:')
    print('准确率:', val_acc)
    print('F1分数:', val_f1)
    print('召回率:', val_recall)
    print('精确率:', val_precision)

    print('测试集结果:')
    print('准确率:', test_acc)
    print('F1分数:', test_f1)
    print('召回率:', test_recall)
    print('精确率:', test_precision)

    results = []
    test_aucroc = roc_auc_score(test_y, test_pred)

    # 获取概率值而不是预测类别
    y_prob = cls.predict_proba(test_x)[:, 1]  # 选择正类（类别 1）的概率
    
    # 计算精度-召回率曲线
    precision1, recall1, thresholds = precision_recall_curve(test_y, y_prob)
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

    print("测试集详细结果:",
          "准确率= {:.4f}".format(test_acc),
          "精确率= {:.4f}".format(test_precision),
          "召回率= {:.4f}".format(test_recall),
          "F1分数= {:.4f}".format(test_f1),
          "AUCROC= {:.4f}".format(test_aucroc),
          "AUCPR= {:.4f}".format(aucpr),
          "精确率80%时的召回率={:.4f}".format(recall_at_p80),
          "精确率85%时的召回率={:.4f}".format(recall_at_p85),
          "精确率90%时的召回率={:.4f}".format(recall_at_p90)
          )

    # 记录结果
    results.append({
        '种子': args.seed,
        '准确率': test_acc,
        '精确率': test_precision,
        '召回率': test_recall,
        'F1': test_f1,
        'AUCROC': test_aucroc,
        'AUC-PR': aucpr,
        'Recall@P80': recall_at_p80,
        'Recall@P85': recall_at_p85,
        'Recall@P90': recall_at_p90
    })
    
    # 生成表格
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))

    # 保存结果到CSV
    results_df.to_csv('sgbot_MGTAB.csv', mode='a', header=not os.path.exists('sgbot_MGTAB.csv'), index=False)
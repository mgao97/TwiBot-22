import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import argparse

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/dev/shm/mgtab/', help='Dataset path for MGATB')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)

# === 数据加载 ===
features = torch.load(os.path.join(args.dataset_path, 'features.pt'))  # [N, D]
labels = torch.load(os.path.join(args.dataset_path, 'labels_bot.pt'))  # [N]

# === 数据集类 ===
class MGATBDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 构造数据集
dataset = MGATBDataset(features, labels)
num_total = len(dataset)
num_train = int(0.7 * num_total)
num_val = int(0.2 * num_total)
num_test = num_total - num_train - num_val

train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test])

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

# === 模型定义 ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.model(x)

input_dim = features.size(1)
model = MLP(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# === 评估函数 ===
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    precision1, recall1, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall1, precision1)
    # 精确率为 80%、85%、90% 时的召回率
    recall_at_p80 = next((r for p, r in zip(precision1, recall1) if p >= 0.8), 0)
    recall_at_p85 = next((r for p, r in zip(precision1, recall1) if p >= 0.85), 0)
    recall_at_p90 = next((r for p, r in zip(precision1, recall1) if p >= 0.9), 0)

    return acc, f1, auc_roc, precision, recall, auc_pr, recall_at_p80, recall_at_p85, recall_at_p90

# === 训练 ===
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"\n[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}")
    acc, f1, auc_roc, precision, recall, auc_pr, recall_at_p80, recall_at_p85, recall_at_p90 = evaluate(model, val_loader)
    print(f"[Validation] ACC: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc_roc:.4f}, PR-AUC: {auc_pr:.4f}")

# === 测试 ===
acc, f1, auc_roc, precision, recall, auc_pr, recall_at_p80, recall_at_p85, recall_at_p90 = evaluate(model, test_loader)
print("\n[Test Result]")
print(f"ACC: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc_roc:.4f}, PR-AUC: {auc_pr:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

# === 保存到 CSV ===
results = pd.DataFrame([{
    'seed': args.seed,
    'Accuracy': acc,
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'AUCROC': auc_roc,
    'AUC-PR': auc_pr,
    'Recall@P80': recall_at_p80, 
    'Recall@P85': recall_at_p85,
    'Recall@P90': recall_at_p90
}])

# 打印 Markdown 表格（可选）
print(results.to_markdown(index=False))

# 保存到 CSV（如果文件不存在就写 header）
results.to_csv('T5_MGTAB.csv', mode='a', header=not os.path.exists('T5_MGTAB.csv'), index=False)
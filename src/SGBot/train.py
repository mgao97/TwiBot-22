import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from argparse import ArgumentParser
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve,auc,roc_auc_score
from tqdm import tqdm

import os
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument('--dataset', type=str,default='Twibot-20')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

dataset = args.dataset

assert dataset in ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']
if dataset == 'Twibot-22':
    path = '/dev/shm/twi22/data/'
elif dataset == 'Twibot-20':
    path = '/dev/shm/twi20/data/'
split = pd.read_csv(path+'split.csv'.format(dataset))
idx = json.load(open(path+'tmp/{}/idx.json'.format(dataset)))
idx = {item: index for index, item in enumerate(idx)}
features = np.load(path+'tmp/{}/features.npy'.format(dataset), allow_pickle=True)
labels = np.load(path+'tmp/{}/labels.npy'.format(dataset))

train_idx = []
val_idx = []
test_idx = []

for index, item in tqdm(split.iterrows(), ncols=0):
    try:
        if item['split'] == 'train':
            train_idx.append(idx[item['id']])
        if item['split'] == 'val' or item['split'] == 'valid':
            val_idx.append(idx[item['id']])
        if item['split'] == 'test':
            test_idx.append(idx[item['id']])
    except KeyError:
        continue

print('loading done')

print(len(train_idx))
print(len(val_idx))
print(len(test_idx))

if __name__ == '__main__':
    train_x = features[train_idx]
    train_y = labels[train_idx]
    val_x = features[val_idx]
    val_y = labels[val_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]
    print('training......')
    cls = RandomForestClassifier(n_estimators=100)
    cls.fit(train_x, train_y)
    print('done.')

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

    print('validation:')
    print('acc:', val_acc)
    print('f1:', val_f1)
    print('recall:', val_recall)
    print('precision:', val_precision)

    print('test:')
    print('acc:', test_acc)
    print('f1:', test_f1)
    print('recall:', test_recall)
    print('precision:', test_precision)

    results = []
    test_aucroc = roc_auc_score(test_y, test_pred)

    # 获取概率值而不是预测类别
    y_prob = cls.predict_proba(test_x)[:, 1]  # 选择正类（类别 1）的概率
    sorted_indices = np.argsort(y_prob)
    y_pred_sorted = y_prob[sorted_indices]
    # 计算精度-召回率曲线
    precision1, recall1, thresholds = precision_recall_curve(test_y, y_pred_sorted)
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
                "test_accuracy= {:.4f}".format(test_acc),
                "precision= {:.4f}".format(test_precision),
                "recall= {:.4f}".format(test_recall),
                "f1_score= {:.4f}".format(test_f1),
                "aucroc= {:.4f}".format(test_aucroc),
                "aucpr= {:.4f}".format(aucpr),
                "recall at precision 80%={:.4f}".format(recall_at_p80),
                "recall at precision 85%={:.4f}".format(recall_at_p85),
                "recall at precision 90%={:.4f}".format(recall_at_p90)
                )

    # 记录结果
    results.append({
        'seed': args.seed,
        'Accuracy': test_acc,
        'Precision': test_precision,
        'Recall': test_recall,
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
    results_df.to_csv('SGBot_twi20.csv', mode='a', header=not os.path.exists(f'SGBot_twi20.csv'), index=False)


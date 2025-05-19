import os
import json
import ijson
import argparse
import pandas as pd
import numpy as np
from feature_engineering import feature_preprocess
from feature_twibot22 import preprocess
from feature_supplement import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve,auc
import ijson
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-22', help='Choose the dataset.')
parser.add_argument('--seed', type=int, default=97, help='Random seed.')
arg = parser.parse_args()
DATASET = arg.dataset

if DATASET == 'Twibot-22':
    path = '/dev/shm/twi22/data/'
    label = pd.read_csv(path+'label.csv')
    split = pd.read_csv(path+'split.csv')
    user = list(ijson.items(open(path+'user.json', 'r'), 'item'))
    author = []
    tid = []
    tweet = []
    

    for i in tqdm(range(9)):
        with open(path + f'tweet_{i}.json', 'r') as f:
            items = ijson.items(f, 'item')
            for item in items:
                tweet.append(item.get('text'))
                tid.append(item.get('id'))
                author.append(item.get('author_id'))

    # for i in range(9):
    #     tweet = tweet + list(ijson.items(open(path+'tweet_' + str(i) + '.json', 'r'), 'item.text'))
    #     tid = tid + list(ijson.items(open(path+'tweet_' + str(i) + '.json', 'r'), 'item.id'))
    #     author = author + list(ijson.items(open(path+'tweet_' + str(i) + '.json', 'r'), 'item.author_id'))
    
    # id_tweet = dict()
    # id_map = dict()
    # num_user = len(user)
    # for i in range(num_user):
    #     id_map[user[i]['id']] = i
    # for i in range(len(tid)):
    #     if id_map[author[i]] in id_tweet.keys():
    #         id_tweet[id_map[author[i]]].append(tweet[i])
    #     else:
    #         id_tweet[id_map[author[i]]] = [tweet[i]]

    
    edge = pd.read_csv(path+'edge.csv')
    id_map = {user[i]['id']: i for i in range(len(user))}
    id_tweet = defaultdict(list)

    for i in range(len(tid)):
        author_id = author[i]
        if author_id in id_map:
            user_idx = id_map[author_id]
            id_tweet[user_idx].append(tweet[i])
    label_order = np.array(label['label'].values)
    split_order = np.array(split['split'].values)
    for i in range(len(user)):
        label_order[id_map[label['id'][i]]] = label['label'][i]
        split_order[id_map[split['id'][i]]] = split['split'][i]
    y = (label_order == 'bot').astype(int)
    train_split = split_order[0: len(user)] == 'train'
    val_split = split_order[0: len(user)] == 'valid'
    test_split = split_order[0: len(user)] == 'test'
    train_set = np.where(split_order == 'train')[0]
    val_set = np.where(split_order == 'valid')[0]
    test_set = np.where(split_order == 'test')[0]
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    if os.path.exists(path+'feature_matrix_Twibot-22.csv'):
        X = pd.read_csv(path+'feature_matrix_Twibot-22.csv').values
    else:
        X = preprocess(user, tid, author, edge, id_tweet)
    print(f"X shape: {X.shape}")
    for i in range(X.shape[0]):
        X[i][np.isnan(X[i])] = np.nanmean(X[i])
else:
    path = '/dev/shm/twi20/data/'
    node = json.load(open(path+'node.json', 'r'))
    label = pd.read_csv(path+'label.csv')
    split = pd.read_csv(path+'split.csv')
    edge = pd.read_csv(path+'edge.csv')
    id_map = dict()
    for i in range(len(node)):
        id_map[node[i]['id']] = i
    num_user = label.shape[0]
    label_order = np.array(label['label'].values)
    split_order = np.array(split['split'].values)
    for i in range(num_user):
        label_order[id_map[label['id'][i]]] = label['label'][i]
        split_order[id_map[split['id'][i]]] = split['split'][i]
    y = (label_order == 'bot').astype(int)
    train_split = split_order[0: num_user] == 'train'
    val_split = split_order[0: num_user] == 'val'
    test_split = split_order[0: num_user] == 'test'
    train_set = np.where(split_order == 'train')[0]
    val_set = np.where(split_order == 'val')[0]
    test_set = np.where(split_order == 'test')[0]
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    if os.path.exists(path+'feature_matrix_' + DATASET + '.csv'):
        X = pd.read_csv(path+'feature_matrix_' + DATASET + '.csv').values[0: num_user]
    else:
        X = feature_preprocess(node, edge, DATASET)[0: num_user]
        # X = preprocessing(node, edge, DATASET)[0: num_user]
    print(f"X shape: {X.shape}")
    for i in range(X.shape[0]):
        X[i][np.isnan(X[i])] = np.nanmean(X[i])

acc_list = []
precision_list = []
recall_list = []
f1_list = []
aucroc_list = []
aucpr_list = []
results = []

for seed in (97, 815, 1149, 945371, 123456):
    print(f"Random seed: {seed}")
    clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=seed)
    clf.fit(X[train_set], y[train_set])
    test_result = clf.predict(X[test_set])
    y_prob = clf.predict_proba(X[test_set])[:, 1]

    acc = accuracy_score(y[test_set], test_result)
    precision = precision_score(y[test_set], test_result)
    recall = recall_score(y[test_set], test_result)
    f1 = f1_score(y[test_set], test_result)
    aucroc = roc_auc_score(y[test_set], test_result)

    # PR curve
    precision1, recall1, thresholds = precision_recall_curve(y[test_set], y_prob)
    aucpr = auc(recall1, precision1)

    # 精确率阈值下的召回率
    def recall_at_precision_threshold(target_p):
        for p, r in zip(precision1, recall1):
            if p >= target_p:
                return r
        return 0.0
    
    recall_at_p80 = recall_at_precision_threshold(0.8)
    recall_at_p85 = recall_at_precision_threshold(0.85)
    recall_at_p90 = recall_at_precision_threshold(0.9)

        
    print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
    print('*'*100)

    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, aucroc: {aucroc:.4f}, aucpr: {aucpr:.4f}")
    print("*" * 80)

    # 保存每轮的结果
    results.append({
        'seed': seed,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUCROC': aucroc,
        'AUC-PR': aucpr,
        'Recall@P80': recall_at_p80,
        'Recall@P85': recall_at_p85,
        'Recall@P90': recall_at_p90
    })

# 生成表格
results_df = pd.DataFrame(results)
print(results_df.to_markdown(index=False))

# 保存结果到CSV
results_df.to_csv('FriendBot.csv', mode='a', header=not os.path.exists(f'FriendBot.csv'), index=False)



# clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=100)
# clf.fit(X[train_set], y[train_set])
# test_result = clf.predict(X[test_set])
# print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
# acc.append(accuracy_score(y[test_set], test_result))
# precision.append(precision_score(y[test_set], test_result))
# recall.append(recall_score(y[test_set], test_result))
# f1.append(f1_score(y[test_set], test_result))
# auc.append(roc_auc_score(y[test_set], test_result))

# clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=200)
# clf.fit(X[train_set], y[train_set])
# test_result = clf.predict(X[test_set])
# print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
# acc.append(accuracy_score(y[test_set], test_result))
# precision.append(precision_score(y[test_set], test_result))
# recall.append(recall_score(y[test_set], test_result))
# f1.append(f1_score(y[test_set], test_result))
# auc.append(roc_auc_score(y[test_set], test_result))

# clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=300)
# clf.fit(X[train_set], y[train_set])
# test_result = clf.predict(X[test_set])
# print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
# acc.append(accuracy_score(y[test_set], test_result))
# precision.append(precision_score(y[test_set], test_result))
# recall.append(recall_score(y[test_set], test_result))
# f1.append(f1_score(y[test_set], test_result))
# auc.append(roc_auc_score(y[test_set], test_result))

# clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=400)
# clf.fit(X[train_set], y[train_set])
# test_result = clf.predict(X[test_set])
# print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
# acc.append(accuracy_score(y[test_set], test_result))
# precision.append(precision_score(y[test_set], test_result))
# recall.append(recall_score(y[test_set], test_result))
# f1.append(f1_score(y[test_set], test_result))
# auc.append(roc_auc_score(y[test_set], test_result))


# results = pd.DataFrame({'seed':arg.seed,'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'aucroc': aucroc, 'aucpr': aucpr})
# results.to_csv('FriendBot.csv', index=False)

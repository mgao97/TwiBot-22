from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn.functional as func


def null_metrics():
    return {
        'acc': 0.0,
        'f1-score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        # 'mcc': 0.0,
        'roc-auc': 0.0,
        'pr-auc': 0.0,
        'Recall@P80': 0.0,
        'Recall@P85': 0.0,
        'Recall@P90': 0.0
    }


def calc_metrics(y, pred):
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog
    pred = func.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred, dim=-1)
    pred_score = pred[:, -1]
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)

    # 计算精确率为80%时的召回率
    recall_at_p80 = 0
    for pi, ri in zip(precision, recall):
        if pi >= 0.8:
            recall_at_p80 = ri
            break

    # 计算精确率为85%时的召回率
    recall_at_p85 = 0
    for pi, ri in zip(precision, recall):
        if pi >= 0.85:
            recall_at_p85 = ri
            break

    # 计算精确率为90%时的召回率
    recall_at_p90 = 0
    for pi, ri in zip(precision, recall):
        if pi >= 0.9:
            recall_at_p90 = ri
            break

    metrics = {
        
        'acc': accuracy_score(y, pred_label),
        'precision': precision_score(y, pred_label),
        'recall': recall_score(y, pred_label),
        'f1-score': f1_score(y, pred_label),
        # 'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision),
        'Recall@P80': recall_at_p80,
        'Recall@P85': recall_at_p85,
        'Recall@P90': recall_at_p90
    }

    plog = ''
    for key in ['acc', 'f1-score']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog


def is_better(now, pre):
    if now['acc'] != pre['acc']:
        return now['acc'] > pre['acc']
    if now['f1-score'] != pre['f1-score']:
        return now['f1-score'] > pre['f1-score']
    # if now['mcc'] != pre['mcc']:
    #     return now['mcc'] > pre['mcc']
    if now['pr-auc'] != pre['pr-auc']:
        return now['pr-auc'] > pre['pr-auc']
    if now['roc-auc'] != pre['roc-auc']:
        return now['roc-auc'] > pre['roc-auc']
    if now['precision'] != pre['precision']:
        return now['precision'] > pre['precision']
    if now['recall'] != pre['recall']:
        return now['recall'] > pre['recall']
    return False

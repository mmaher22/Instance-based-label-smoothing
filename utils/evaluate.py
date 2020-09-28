import numpy as np
from netcal.metrics import ECE, MCE, ACE
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def eval(y_preds, y_true, num_classes = 10, bins = 15):
    try:
        y_preds = y_preds.cpu().numpy(); y_true = y_true.cpu().numpy()
    except:
        y_preds = y_preds.numpy(); y_true = y_true.numpy()

    ece_score, ace_score, mce_score = eval_cal(y_preds, y_true, bins)
    cw_ece, cw_ace, cw_mce = eval_cal_cw(y_preds, y_true, num_classes, bins)
    l1, l2 = eval_loss(y_preds, y_true)
    acc = eval_acc(y_preds, y_true)
    return ece_score, ace_score, mce_score, cw_ece, cw_ace, cw_mce, l1, l2, acc

def eval_cal(y_preds, y_true, bins = 15):
    # Calibration Metrics
    ece = ECE(bins); ace = ACE(bins); mce = MCE(bins)
    ece_score = ece.measure(y_preds, y_true)
    ace_score = ace.measure(y_preds, y_true)
    mce_score = mce.measure(y_preds, y_true)
    return ece_score, ace_score, mce_score

def eval_cal_cw(y_preds, y_true, num_classes = 10, bins = 15):
    # Class-Wise Metrics
    ece_lst = []; ace_lst = []; mce_lst = []
    for cl in range(num_classes):
        inds = np.where(y_true == cl)[0]
        e, a, m = eval_cal(y_preds[inds, :], y_true[inds])
        ece_lst.append(e); ace_lst.append(a); mce_lst.append(m);
    return np.average(ece_lst), np.average(ace_lst), np.average(mce_lst)

def eval_loss(y_preds, y_true):
    # Proper Loss Metrics
    try:
        loss1 = log_loss(y_true, y_preds, eps = 1e-15)
    except:
        loss1 = 1e15
    try:
        loss2 = brier_score_loss(np.ones(len(y_true)), np.choose(y_true, y_preds.T), pos_label = 1)
    except:
        loss2 = 0
    return loss1, loss2

def eval_acc(y_preds, y_true):
    # Accuracy
    return accuracy_score(y_true, np.argmax(y_preds, axis = -1))
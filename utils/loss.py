## Imports
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
from netcal.metrics import ECE
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)

#####################################################################################################################

#### Cross Entropy with Label Smoothing
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing = 0.1, reduction = 'mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

#####################################################################################################################
#### Cross Entropy with Class-based Label Smoothing 
class CLabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, class_avg, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(CLabelSmoothingCrossEntropyLoss, self).__init__()
        self.class_avg = class_avg
        self.temperature = temperature
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target):
        self.class_avg = F.softmax(self.class_avg / self.temperature, dim = -1)
        sm = torch.zeros(len(target), self.num_classes).to(device)
        sums = torch.sum(self.class_avg, axis = -1)
        sums -= torch.diag(self.class_avg)
        for i in range(len(target)):
            t = target[i].item()
            smoothing_factor = self.smoothing
            sm[i, t] = 1 - smoothing_factor
            for j in range(self.num_classes):
                if j == t:
                    continue
                else:
                    sm[i, j] = smoothing_factor * (1 - self.class_avg[t, j]) / sums[t]
        logprobs = F.log_softmax(x, dim=-1)
        loss = -(logprobs * sm).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

#####################################################################################################################

 #### Cross Entropy with Instance-based Label Smoothing (Version 1) - Varying the smoothing factor proportional to the teacher confidence - Same for all incorrect
class ILabelSmoothingCrossEntropyLoss1(nn.Module):
    def __init__(self, orig_net, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(ILabelSmoothingCrossEntropyLoss1, self).__init__()
        self.orig_net = orig_net
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False).to(device)
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target, inputs):
        with torch.no_grad():
            sm = self.orig_net(inputs) / self.temperature
        sm = F.softmax(sm, dim=-1)
        #sums = torch.sum(sm, dim=-1)
        #smoothing_factor = self.smoothing * torch.max(sm, dim = -1)[0] / sums
        smoothing_factor = self.smoothing * sm.gather(dim=-1, index=target.unsqueeze(1))[:,0]# / sums
        sums -= sm.gather(dim=-1, index=target.unsqueeze(1))[:,0]
        sm[:,:] = smoothing_factor.unsqueeze(-1) / (self.num_classes - 1)
        sm[torch.arange(0, len(sm)).long(), target.long()] = 1 - smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -(logprobs * sm).sum(dim=-1)
        loss =  smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

    #####################################################################################################################

 #### Cross Entropy with Instance-based Label Smoothing (Version 11) -  Varying the smoothing factor inversely  proportional to the teacher confidence - Same for all incorrect
class ILabelSmoothingCrossEntropyLoss11(nn.Module):
    def __init__(self, orig_net, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(ILabelSmoothingCrossEntropyLoss11, self).__init__()
        self.orig_net = orig_net
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False).to(device)
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target, inputs):
        with torch.no_grad():
            sm = self.orig_net(inputs) / self.temperature
        sm = F.softmax(sm, dim=-1)
        sums = torch.sum(sm, dim=-1)
        #smoothing_factor = self.smoothing * (sums - torch.max(sm, dim = -1)[0] ) / sums
        smoothing_factor = self.smoothing * (sums - sm.gather(dim=-1, index=target.unsqueeze(1))[:,0]) #/ sums
        #sums -= sm.gather(dim=-1, index=target.unsqueeze(1))[:,0]
        sm[:,:] = smoothing_factor.unsqueeze(-1) / (self.num_classes - 1)
        sm[torch.arange(0, len(sm)).long(), target.long()] = 1 - smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -(logprobs * sm).sum(dim=-1)
        loss =  smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

#####################################################################################################################

#### Cross Entropy with Instance-based Label Smoothing (Version 2) - Constant smoothing factor but varying to each incorrect according to its confidence
class ILabelSmoothingCrossEntropyLoss2(nn.Module):
    def __init__(self, orig_net, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(ILabelSmoothingCrossEntropyLoss2, self).__init__()
        self.orig_net = orig_net
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False).to(device)
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target, inputs):
        with torch.no_grad():
            sm = self.orig_net(inputs) / self.temperature
        sm = F.softmax(sm, dim=-1)
        sums = torch.sum(sm, dim=-1)
        smoothing_factor = self.smoothing
        sums -= sm.gather(dim=-1, index=target.unsqueeze(-1))[:,0]
        sm = self.smoothing * sm / sums.unsqueeze(-1)
        sm[torch.arange(0, len(sm)).long(), target.long()] = 1 - smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -(logprobs * sm).sum(dim=-1)
        loss =  smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

#####################################################################################################################

#### Cross Entropy with Instance-based Label Smoothing12 (Version 1+2) - Versions: 1 + 2
class ILabelSmoothingCrossEntropyLoss12(nn.Module):
    def __init__(self, orig_net, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(ILabelSmoothingCrossEntropyLoss12, self).__init__()
        self.orig_net = orig_net
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False).to(device)
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target, inputs):
        with torch.no_grad():
            sm = self.orig_net(inputs) / self.temperature
        sm = F.softmax(sm, dim=-1)
        sums = torch.sum(sm, dim=-1)
        #smoothing_factor = self.smoothing * torch.max(sm, dim = -1)[0] / sums
        smoothing_factor = self.smoothing * sm.gather(dim=-1, index=target.unsqueeze(1))[:,0] / sums
        sums -= sm.gather(dim=-1, index=target.unsqueeze(-1))[:,0]
        sm = self.smoothing * sm / sums.unsqueeze(-1)
        sm[torch.arange(0, len(sm)).long(), target.long()] = 1 - smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -(logprobs * sm).sum(dim=-1)
        loss =  smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

#####################################################################################################################

#### Cross Entropy with Instance-based Label Smoothing (Version 11 + 2) - Versions: 11 + 2
class ILabelSmoothingCrossEntropyLoss112(nn.Module):
    def __init__(self, orig_net, temperature = 1, smoothing = 0.1, num_classes = 2, reduction = 'mean'):
        super(ILabelSmoothingCrossEntropyLoss112, self).__init__()
        self.orig_net = orig_net
        self.temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False).to(device)
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes
    def forward(self, x, target, inputs):
        with torch.no_grad():
            sm = self.orig_net(inputs) / self.temperature
        sm = F.softmax(sm, dim=-1)
        sums = torch.sum(sm, dim=-1)
        smoothing_factor = self.smoothing * (sums - sm.gather(dim=-1, index=target.unsqueeze(1))[:,0]) #/ sums
        sums -= sm.gather(dim=-1, index=target.unsqueeze(-1))[:,0]
        sm = self.smoothing * sm / sums.unsqueeze(-1)
        sm[torch.arange(0, len(sm)).long(), target.long()] = 1 - smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        smooth_loss = -(logprobs * sm).sum(dim=-1)
        loss =  smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
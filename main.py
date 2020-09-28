# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:18:22 2020
@author: Mohamed Maher
"""
# Other imports
import os
import gc
import json
import copy
import argparse
import random as rn
import numpy as np
import pandas as pd
from utils.utils import progress_bar
# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
# Datasets imports
from datasets.data_loader import DatasetLoader
# Models Imports
from models.lenet import lenet
from models.convnet import convnet
from models.resnet_wide import resnet32_wide
from models.densenet import densenet161, densenet40
from models.resnet import resnet50, resnet110, resnet152
from models.resnet_sd import resnet110_SD, resnet152_SD
# Losses Imports
from utils.loss import LabelSmoothingCrossEntropyLoss, CLabelSmoothingCrossEntropyLoss
from utils.loss import ILabelSmoothingCrossEntropyLoss1, ILabelSmoothingCrossEntropyLoss11, ILabelSmoothingCrossEntropyLoss2
from utils.loss import ILabelSmoothingCrossEntropyLoss12, ILabelSmoothingCrossEntropyLoss112
# Evaluation Imports
from utils.evaluate import eval
# Temperature Scaling
from utils.utils import TemperatureScaling

# Initializations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
smooth_factors = [0.01, 0.05, 0.1, 0.15, 0.2]
temp_factors = [1, 2, 4, 8, 16]

# Parser Arguments
parser = argparse.ArgumentParser(description='PyTorch Instance-based label smoothing evaluation')
parser.add_argument('--dataset', default='cifar10', type=str, help='Name of the dataset to be used, one of {cifar10, cifar100, svhn, imagenet, cars, birds}' )
parser.add_argument('--model', default='resnet50', type=str, help='Name of the model architecture, one of {lenet, convnet, resnet32_wide, densenet40, densenet161, resnet50, resnet110, resnet152, resnet110_SD, resnet152_SD}' )
parser.add_argument('--loss', default='ce', type=str, help='Name of the loss function, one of {ce, ls-ce, cls-ce, ils1-ce, ils11-ce, ils2-ce, ils12-ce, ils112-ce}' )
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='no of epochs')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--early', default=1000000, type=int, help='early stopping')
parser.add_argument('--temperature', default=False, type=bool, help='Use temperature scaling or not')
parser.add_argument('-scheduler', help='delimited list int input of the epochs at which scheduler will change the learning rate', type=str, default = '80,150')
parser.add_argument('-orig_path', help='path of the network trained without label smoothing', type=str, default = 'None')
args = parser.parse_args()
args.schedule = [int(item) for item in args.scheduler.split(',')]
#Set Random Seeds
rn.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#####################################################################################################################
# Model
def create_model(model):
    print('==> 2. Building the Model...')
    if model == 'lenet':
        net = lenet(dataset.num_classes, dataset.img_channels).to(device)
    elif model == 'convnet':
        net = convnet(dataset.num_classes, dataset.img_channels).to(device)
    elif model == 'resnet32_wide':
        net = resnet32_wide(dataset.num_classes, dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'densenet40':
        net = densenet40(num_classes = dataset.num_classes, in_channels = dataset.img_channels).to(device)
    elif model == 'densenet161':
        net = densenet161(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'resnet50':
        net = resnet50(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'resnet110':
        net = resnet110(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'resnet152':
        net = resnet152(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'resnet110_SD':
        net = resnet110_SD(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    elif model == 'resnet152_SD':
        net = resnet152_SD(num_classes = dataset.num_classes, in_channels = dataset.img_channels, pretrained = dataset.require_pretrained).to(device)
    return net

#####################################################################################################################
 # Loss Function
def create_loss(loss, smooth = 0.1, temp = 4, num_classes = 10, orig_net = None, class_avg = None):
        print('==> 3. Choosing Loss Function...')
        if loss == 'ce':
            criterion = nn.CrossEntropyLoss(reduction = 'mean')
        elif loss == 'ls-ce':
            criterion = LabelSmoothingCrossEntropyLoss(smoothing = smooth, reduction = 'mean')
        elif loss == 'cls-ce':
            criterion = CLabelSmoothingCrossEntropyLoss(class_avg = class_avg, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        elif loss == 'ils1-ce':
            criterion = ILabelSmoothingCrossEntropyLoss1(orig_net = orig_net, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        elif loss == 'ils11-ce':
            criterion = ILabelSmoothingCrossEntropyLoss11(orig_net = orig_net, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        elif loss == 'ils2-ce':
            criterion = ILabelSmoothingCrossEntropyLoss2(orig_net = orig_net, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        elif loss == 'ils12-ce':
            criterion = ILabelSmoothingCrossEntropyLoss12(orig_net = orig_net, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        elif loss == 'ils112-ce':
            criterion = ILabelSmoothingCrossEntropyLoss112(orig_net = orig_net, smoothing = smooth, num_classes = num_classes, temperature = temp, reduction = 'mean')
        return criterion

#####################################################################################################################
class Master():
    def __init__(self, dataset, epochs = args.epochs, model = args.model, lr = args.lr, loss = args.loss, decay = args.decay, temperature = False,
                        temp_factors = temp_factors, smooth_factors = smooth_factors, schedule = args.schedule, orig_net_path = None, seed = 0, stop = args.early):
        self.best_net = None; self.best_loss = 1e12; self.best_temp = 1; self.best_smooth = 0; self.dataset = dataset; self.loss = loss; self.stop = stop
        self.epochs = epochs; self.t = 1; self.f = 0; self.temperature = temperature; self.seed = seed; self.schedule = schedule; self.lr = lr; self.decay = decay
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'mean')

        if self.loss == 'ce':
            self.net = create_model(model)
            self.criterion = create_loss(loss)
            self.create_optimizer()
            self.trainer_main()
        elif self.loss == 'ls-ce':
            for f in smooth_factors:
                self.net = create_model(model); self.f = f
                self.criterion = create_loss(loss, smooth = f)
                self.create_optimizer()
                self.trainer_main()
        elif self.loss == 'cls-ce':
            orig_net = create_model(model)
            orig_net.load_state_dict(torch.load(orig_net_path))
            orig_net.eval()
            class_avg = self.get_class_avg(orig_net)
            for f in smooth_factors:
                for t in temp_factors:
                    self.net = create_model(model); self.t = t; self.f = f
                    self.criterion = create_loss(loss, smooth = f, temp = t, class_avg = class_avg)
                    self.create_optimizer()
                    self.trainer_main()
        else:
            orig_net = create_model(model)
            orig_net.load_state_dict(torch.load(orig_net_path))
            orig_net.eval()
            for f in smooth_factors:
                for t in temp_factors:
                    self.net = create_model(model); self.t = t; self.f = f
                    self.criterion = create_loss(loss, smooth = f, temp = t, orig_net = orig_net)
                    self.create_optimizer()
                    self.trainer_main()

        self.evaluate()

    def get_class_avg(self, orig_net):
        class_avg = np.zeros((self.dataset.num_classes , self.dataset.num_classes ))
        samples_no = np.zeros((self.dataset.num_classes, 1))
        # Find Average of logits per class
        for instances, labels in self.dataset.validloader:
            instances, labels = instances.to(device), labels.to(device)
            with torch.no_grad():
                #print(type(instances), device)
                outputs = orig_net(instances)
                if device == 'cpu':
                    smoothing_outputs = outputs.numpy()
                else:
                    smoothing_outputs = outputs.cpu().numpy()
            for i, label in enumerate(labels):
                class_avg[int(label.item()), :] += smoothing_outputs[i, :]
                samples_no[label.item()] += 1
        for i in range(self.dataset.num_classes):
            class_avg[i, :] /= samples_no[i]
        return torch.from_numpy(class_avg).float().to(device)

        
#####################################################################################################################
    #Optimizer
    def create_optimizer(self):
        print('==> 4. Defining the Optimizer...')
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.decay, nesterov = True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.schedule, gamma = 0.1)

#####################################################################################################################
    # Training        
    def trainer_main(self):
        print('==> 5. Training the Model...')
        model_save_dir = dir2 + self.loss + '_' + str(self.seed) + '_False_' + str(self.t) + '_' + str(self.f) + '.pt'
        self.best_tmp_net = None; self.best_tmp_loss = 1e9
        if self.temperature:
            print('Loading the model to apply temperature scaling...')
            self.best_tmp_net.load_state_dict( torch.load(model_save_dir) )
            temp_model = TemperatureScaling(self.net, solver = 'L-BFGS-B')
            temp = temp_model.fit(dataset.validloader, seed = self.seed)
            self.best_tmp_net.temperature = torch.nn.Parameter(torch.ones(1) * temp, requires_grad=False).to(device)
        else:
            model_save_dir = dir2 + self.loss + '_' + str(self.seed) + '_' + str(self.temperature) + '_' + str(self.t) + '_' + str(self.f) + '.pt'
            if os.path.isfile(model_save_dir):
                self.best_tmp_net.load_state_dict( torch.load(model_save_dir) )
            else:
                self.early = 0
                for epoch in range(1, self.epochs + 1):
                    self.early += 1
                    self.train(epoch)
                    self.validate()
                    if self.early >= self.stop:
                        break
        model_save_dir = dir2 + self.loss + '_' + str(self.seed) + '_' + str(self.temperature) + '_' + str(self.t) + '_' + str(self.f) + '.pt'
        torch.save(self.best_tmp_net.state_dict(), model_save_dir)

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0; correct = 0; total = 0; tt = 0
        for batch_idx, (inputs, targets) in enumerate(self.dataset.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            try:
                loss = self.criterion(outputs, targets)
            except:
                loss = self.criterion(outputs, targets, inputs)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            _, predicted = outputs.max(-1)
            total += targets.size(0)
            tt += 1
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(self.dataset.trainloader), 'Loss: %.5f | Acc: %.3f%%' % (train_loss / tt, 100. * correct / total) )
        self.scheduler.step()

    def validate(self):
        valid_loss = 0; total = 0
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.net(inputs)
                loss = self.ce_loss(outputs, targets)
                valid_loss += loss.item()
                total += 1#targets.size(0)
        valid_loss /= total
        #print('Validation Loss:', valid_loss)
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss; self.best_temp = self.t; self.best_smooth = self.f
            self.best_net = copy.deepcopy(self.net)
        if valid_loss < self.best_tmp_loss:
            self.best_tmp_loss = valid_loss; self.early = 0
            self.best_tmp_net = copy.deepcopy(self.net)
#####################################################################################################################
    # Evaluation of Results
    def evaluate(self):
        print('==> 6. Evaluating the Metrics...')
        # Training Set
        tr_loss = 0; total = 0
        self.best_net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.net(inputs)
                loss = self.ce_loss(outputs, targets)
                tr_loss += loss.item()
                total += targets.size(0)
        tr_loss /= total
        self.tr_loss = tr_loss

        # Testing Set
        self.best_net.eval()
        y_preds = []; y_true = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = F.softmax(self.net(inputs), dim = -1)
                y_preds.append(outputs); y_true.append(targets)
        y_preds = torch.cat(y_preds, dim = 0).to(device)
        y_true = torch.cat(y_true, dim = 0).to(device)
        self.ece_score, self.ace_score, self.mce_score, self.cw_ece, self.cw_ace, self.cw_mce, self.te_logloss, self.te_brierloss, self.te_acc = eval(y_preds, y_true)

#####################################################################################################################
# Saving Results
def save_results(master, args = args):
    print('==> 7. Saving...')
    # Argument Parser
    with open(dir2 + 'args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    # Model
    torch.save(master.best_net.state_dict(), dir2 + 'best_' + args.loss + '_' + str(args.seed) + '_' + str(args.temperature) + '.pt')
    # Tabular Results
    results_dict = {'dataset':args.dataset, 'model':args.model, 'loss': args.loss, 'tr_loss':master.tr_loss, 'vl_loss':master.best_loss, 'te_loss':master.te_logloss, 'te_brier':master.te_brierloss, 'te_acc':master.te_acc,
                            'te_ece':master.ece_score, 'te_ace':master.ace_score, 'te_mce':master.mce_score, 'te_cw_ece':master.cw_ece, 'te_cw_ace':master.cw_ace, 'te_cw_mce':master.cw_mce,
                            'smooth':master.best_smooth, 'scale':master.best_temp, 'temperature':master.best_net.temperature.item(), 'seed':args.seed}
    print(results_dict, '\n###################################################################################################')
    if os.path.isfile(dir1 + 'all_results.csv'):
        all_results = pd.read_csv(dir1 + 'all_results.csv')
    else:
        all_results = pd.DataFrame(columns = list(results_dict.keys()) )
    all_results = all_results.append(results_dict, ignore_index = True)
    all_results.to_csv(dir1 + 'all_results.csv', index=False, header=True)

##################################################################################################################################################################
# Parser Arguments
args.dataset = 'cars'
args.model = 'resnet50' #{lenet, convnet, resnet32_wide, densenet40, densenet161, resnet50, resnet110, resnet152, resnet110_SD, resnet152_SD}
#args.model = 'convnet'
args.temperature = False
args.lr = 0.001# 0.0001 #0.001 cars
args.decay = 1e-6#0.0001
args.epochs = 250
args.batch_size = 32
args.scheduler = [250]
args.early = 10
losses = ['ce', 'ls-ce', 'cls-ce', 'ils1-ce', 'ils11-ce', 'ils2-ce', 'ils12-ce', 'ils112-ce']
seeds = [333, 22, 42]

for s in seeds:
    args.seed = s
    for l in losses:
        gc.collect()
        # Saving Directories
        dir1 = 'real_results/' 
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        dir2 = 'real_results/'  + args.dataset + '/' + args.model + '/'
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        print('Loss: ', l, '-- Seed: ', s)
        args.loss = l
        # Original path
        if 'cls' in args.loss or 'ils' in args.loss:
            args.orig_path = 'real_results/' +  args.dataset + '/' + args.model + '/best_ce_' + str(args.seed) + '_False.pt'
        else:
            args.orig_path = 'None'
        dir_save = 'real_results/' +  args.dataset + '/' + args.model + '/best_' + args.loss + '_' + str(args.seed) + '_' + str(args.temperature)+ '.pt'
        if os.path.isfile(dir_save):
            print('Model already exists...Closing!')
            continue
        ##################################################################################################################################################################
        # Call Main Functions
        print('==> 1. Loading Dataset...')
        dataset = DatasetLoader(dataset_name = args.dataset, batch_size = args.batch_size, rv = args.seed)
        master = Master(dataset = dataset, epochs = args.epochs, model = args.model, lr = args.lr, loss = args.loss, decay = args.decay, temperature = args.temperature, 
                                    temp_factors = temp_factors, smooth_factors = smooth_factors, schedule = args.schedule, orig_net_path = args.orig_path, seed = args.seed)
        save_results(master, args)
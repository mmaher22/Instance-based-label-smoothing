import os
import time
import math
import torch
import utils
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import models.resnet as resnet
import models.shallownet as net
from torch.autograd import Variable
import models.inception as inception
from utils.evaluate import evaluate_kd
import utils.data_loader as data_loader
from torch.optim.lr_scheduler import StepLR
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser()

def fetch_teacher_outputs(teacher_model, dataloader, params):
    teacher_model.eval()
    teacher_outputs = []
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        output_teacher_batch = teacher_model(data_batch).data.cpu().numpy()
        teacher_outputs.append(output_teacher_batch)
    return teacher_outputs


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_outputs, optimizer, loss_fn_kd, dataloader, metrics, params):
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            output_batch = model(train_batch)
            output_teacher_batch = torch.from_numpy(teacher_outputs[i])
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(async=True)
            output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)
            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer, loss_fn_kd, metrics, params, model_dir, title):
    best_val_loss = 1e9; best_metrics = None
    teacher_model.eval()
    teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, params)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 
    for epoch in range(params.num_epochs):
        scheduler.step()
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train_kd(model, teacher_outputs, optimizer, loss_fn_kd, train_dataloader, metrics, params)
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)
        val_loss = val_metrics['loss']
        is_best = val_loss<=best_val_loss
        if is_best:
            logging.info("----> Found new best log loss")
            best_val_loss = val_loss
            best_json_path = os.path.join(model_dir, title+"_best_validation_metrics.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            best_metrics = val_metrics
    return best_metrics

def get_new_model(args, tmp_scale = True):
    if args.model == 'resnet18':
        return resnet.ResNet18(tmp_scale = tmp_scale, num_classes = args.num_classes)
    elif args.model == 'resnet50':
        return resnet.ResNet50(tmp_scale = tmp_scale, num_classes = args.num_classes)
    elif args.model == 'resnet101':
        return resnet.ResNet101(tmp_scale = tmp_scale, num_classes = args.num_classes)
    elif args.model == 'inceptionv4':
        return inception.inceptionv4(tmp_scale = tmp_scale, num_classes = args.num_classes)
    elif args.model == 'densenet':
        return densenet.DenseNet(tmp_scale = tmp_scale)

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    # Load the parameters from json file
    parser.add_argument('--model_dir', default='label_smoothing')
    parser.add_argument('--restore_file', default='LabelSmoothing.bin')
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--model', default = 'inceptionv4', help='Model to be used')
    args = parser.parse_args()
    json_path = os.path.join('utils/params.json')
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    if params.cuda: torch.cuda.manual_seed(42)
    results_dict = {}
    results_df = pd.DataFrame()
    for trial in range(3):
        main_path = 'files_' + args.model + '_' + args.dataset + '/trial' + str(trial) + '/' 
        dirs = ['CrossEntropy', 'LabelSmoothing', 'InstanceLabelSmoothing', 'CETmpScaling', 'LSTmpScaling', 'ILSTmpScaling']; 
        files = ['CrossEntropy.bin', 'LabelSmoothing.bin', 'Instance_LabelSmoothing_ce_logits.bin', 
                 'CETmpScaling.bin', 'LSTmpScaling.bin', 'ILSTmpScaling.bin']
        for model_dir, restore_file in zip(dirs, files):
            args.restore_file = restore_file
            # Set the logger
            utils.set_logger(os.path.join(main_path, 'distillation.log'))
            # fetch dataloaders, considering full-set vs. sub-set scenarios
            train_dl = data_loader.fetch_dataloader('train', params, args.dataset)
            dev_dl = data_loader.fetch_dataloader('dev', params, args.dataset)
            ### train a 5-layer CNN network as a student network
            if args.dataset == 'fashionmnist':
                input_channels = 1
            else:
                input_channels = 3
            model = net.Net(params, num_classes = args.num_classes, input_channels = input_channels).cuda() if params.cuda else net.Net(params)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            loss_fn_kd = net.loss_fn_kd
            metrics = net.metrics
            ### Specify the pre-trained teacher models for knowledge distillation
            teacher_model = get_new_model(args)
            teacher_checkpoint = main_path + args.restore_file
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model
            utils.load_checkpoint(teacher_checkpoint, teacher_model)
            # Train the model with KD
            logging.info("Model = {}".format(args.model))
            logging.info("Dataset= {}".format(args.dataset))
            logging.info("Trial Number= {}".format(trial))
            # Train the model with KD
            best_metrics = train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd, metrics, params, main_path, model_dir)
            results_dict['model'] = model_dir; results_dict['trial'] = trial
            results_dict['student_loss'] = best_metrics['loss']; results_dict['student_acc'] = best_metrics['accuracy'];
            results_df = results_df.append(results_dict, ignore_index=True)

    results_df.to_csv(main_path + 'distillation_results.csv', index = False)
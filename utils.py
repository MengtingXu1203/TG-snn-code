import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import numpy as np
import random
from models.layers import *
from matplotlib import pyplot as plt
from functions import soft_loss, TET_loss, ordinal_loss
from torch.autograd import grad
from copy import deepcopy


def train(model, device, train_loader, criterion, optimizer, T, num_labels, dvs, args=None):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        if dvs:
            images = images.transpose(0, 1)
        if T == 0:
            outputs = model(images)
            loss = criterion(outputs, labels)
        else:
            images_t = add_dimension(images, T) #[T,B,C,H,W]
            images_t.requires_grad = True
            outputs = model(images_t) ##[B,T,n_class]

            if args.loss=='ce':
                loss = criterion(outputs.mean(1), labels)
            elif args.loss == 'TET':
                loss = TET_loss(outputs, labels, criterion, means=1, lamb=0.05)
            elif args.loss == 'TGloss':
                SM_output = outputs.mean(1)
                fy = F.nll_loss(F.log_softmax(SM_output, dim=1), labels.to(device))
                grad_x = torch.autograd.grad(fy, images_t, retain_graph=True, create_graph=True)[0]  
                temporalwiseloss = torch.sum((grad_x ** 2),dim=[0,2,3,4]).mean()

                ce = criterion(outputs.mean(1), labels)
                loss = ce + 100 * temporalwiseloss
                print("ce:{}, TGloss:{}".format(ce, temporalwiseloss))

        running_loss += loss.item()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.mean(1).cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


def val(model, test_loader, device, T, dvs, norm=None, atk=None,args=None):
    correct = 0
    total = 0
    model.eval()
    grad_norm_sum = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        if T > 0:
            inputs = add_dimension(inputs, T) #[T,B,C,H,W]
        if dvs:
            inputs = inputs.transpose(0, 1)
        if atk is not None:
            inputs, grad_norm = atk(inputs, targets.to(device)) ##[T,B,C,H,W]
            grad_norm_sum += grad_norm
            model.set_simulation_time(T)
        with torch.no_grad():
            if T > 0:
                outputs = model(inputs) ##[B,T,n]
                if args.temporal_test == 'True':
                    attack_t = args.attack_t + 1
                    outputs = outputs[:,:attack_t,:].mean(1)
                else:
                    outputs = outputs.mean(1)

            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    final_acc = 100 * correct / total
    return final_acc, grad_norm_sum



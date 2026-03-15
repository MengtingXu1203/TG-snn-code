import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import attack
import data_loaders
from functions import *
from models import *
from utils import train, val, rat_train, at_train, sr_train
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=2, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='model')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-T','--time',default=8, type=int,metavar='N',help='snn simulation time, set 0 as ANN')
parser.add_argument('-tau','--tau',default=1., type=float,metavar='N',help='leaky constant')
parser.add_argument('-en', '--encode', default='constant', type=str, help='(constant/poisson/glabol)')

# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-wd','--wd',default=5e-4, type=float,help='weight decay')
parser.add_argument('-loss', '--loss', default='ce', type=str, help='(ce/TET/temporalwiseloss)')
parser.add_argument('-m','--train_method',default='natural', type=str ,help='training methods [rattrain,attrain,natural]')

## for temporalwise attack
parser.add_argument('--temporal_test', default='False', type=str, help='whether test on temporal wise or global')
parser.add_argument('--temporal_loss', default='False', type=str, help='whether loss on temporal wise or global')
parser.add_argument('--temporal_grad', default='False', type=str, help='whether grad on temporal wise or global')
parser.add_argument('--attack_t', default=0, type=int, metavar='N',help='attack time in snn simulation')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    dvs = False
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100

    #>>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    log_dir = 'results/'+'%s-checkpoints/'% (args.dataset)+'%s'%(args.model) + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    if 'dvsgesture' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='')
    elif 'dvscifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar(root='')
    elif 'nmnist' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(root='')
    elif 'tinyimagenet' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_tinyimagenet(root='')
    elif 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
    else:
        raise AssertionError("data not supported")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    if 'cnndvs' in args.model.lower():
        model = CNNDVS(args.time, num_labels, args.tau, 2, init_s)
    elif 'vggdvs' in args.model.lower():
        model = VGGDVS(args.model.lower(), args.time, num_labels, znorm, args.tau)
    elif 'vggdvs11' in args.model.lower():
        model = VGGDVS11(args.model.lower(), args.time, num_labels, znorm, args.tau)
    elif 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm, args.tau, args=args)
    elif 'wideresnet' in args.model.lower():
        model = WideResNet(args.model.lower(), args.time, num_labels, znorm, args=args)
    elif 'resnet17' in args.model.lower():
        model = ResNet17(args.time, args.tau, num_labels, znorm)
    elif 'resnet19' in args.model.lower():
        model = ResNet19(args.time, args.tau, num_labels, znorm)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    model.to(device)
    model.poisson = (args.encode.lower() == 'poisson')

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.model
    identifier += '_T[%d]'%(args.time)
    identifier += '_loss[%s]'%(args.loss)
    identifier += '_train[%s]'%(args.train_method)
    identifier += args.suffix

    writer_tensorboard = SummaryWriter(f'results/'+'%s-checkpoints/'%(args.dataset) +'%s'%(args.model) + "/"+ '%s'%(identifier) + "/")

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    
    for epoch in range(args.epochs):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time,num_labels, dvs=dvs,args=args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        
        writer_tensorboard.add_scalar(tag="Train Loss", scalar_value=loss, global_step=epoch)
        writer_tensorboard.add_scalar(tag="Train Accuracy", scalar_value=acc, global_step=epoch)


        tmp,_ = val(model, test_loader, device, args.time, dvs,znorm,atk=None,args=args)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch, args.epochs, tmp))

        writer_tensorboard.add_scalar(tag="Test Accuracy", scalar_value=tmp, global_step=epoch)

        if best_acc < tmp:
            best_acc = tmp
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))
        logger.info('Best Test acc = {:.3f}\t Best epoch={}'.format(best_acc, best_epoch))

    logger.info('Best Test acc={:.3f}'.format(best_acc))
    

if __name__ == "__main__":
    main()
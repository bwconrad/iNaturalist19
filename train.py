import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import json 
import os
from PIL import Image 
import sys
import urllib.request
from torchsummary import summary
import time
import shutil
import pickle
import argparse

import load_dataset
from models.inception3 import *
from lr_finder import LRFinder

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--train_file', type=str, default='./data/train2019.json')
parser.add_argument('--val_file', type=str, default='./data/val2019.json')
parser.add_argument('--test_file', type=str, default='./data/test2019.json')
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--epochs_ft', type=int, default=10)
parser.add_argument('--size', type=int, default=560)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'])
parser.add_argument('--lr', type=int, default=0.0075)
parser.add_argument('--lr_ft', type=int, default=0.00075)
parser.add_argument('--lr_decay', type=int, default=0.94)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--weight_decay', type=int, default=1e-4)
parser.add_argument('--lr_decay_rate', type=int, default=2)
parser.add_argument('--lr_decay_rate_ft', type=int, default=4)
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--ft_clip', type=int, default=4)

args = parser.parse_args()


def train(loader, model, criterion, optimizer, epoch, epochs):
    model.train() # Set model to training mode

    running_loss = 0.0 
    running_corrects = 0.0 

    print('Epoch:{0}/{1}'.format(epoch+1,epochs))
    start = time.time()

    # Train on all batches
    for i, (inputs, ids, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Compute loss
        output = model(inputs)
        _, preds = torch.max(output, 1)
        loss = criterion(output, labels)
        
        # Propagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add to cumulative loss over batches
        running_loss += loss.item() * len(inputs)
        running_corrects += torch.sum(preds == labels.data)

        if i % print_freq == print_freq-1:
            print('[Epoch %d, Batch %5d] loss: %.3f acc: %.5f' %
                (epoch+1, i + 1, running_loss / ((i+1)*len(inputs)),
                 float(running_corrects) / float(((i+1)*len(inputs)))))
            
    print('Epoch %d loss: %.3f acc: %.5f' %
                (epoch+1, running_loss / float(len(loader.dataset)),
                 float(running_corrects) / float(len(loader.dataset))))  
    
    end = time.time()-start
    print('Epoch complete in {:.0f}m {:.0f}s \n'.format(
        end // 60, end % 60))

def test(loader, model, criterion, save=False, mode='Validation'):
    with torch.no_grad():
        model.eval() # Set model to testing mode
        
        test_loss = 0.0 
        correct = 0.0 

        preds = []
        img_ids = []

        print('Testing on {}'.format(mode))
        for inp, img_id, label in loader:
            inp = inp.to(device)
            label = label.to(device)
          
            # Make prediction with model
            output = model(inp)
            _, pred = torch.max(output, 1)
            loss = criterion(output, label)
            
            # Save prediction
            if save:
                img_ids.append(img_id.cpu().numpy().astype(np.int))
                preds.append(pred.cpu().numpy().astype(np.int))

            test_loss += loss.item() * len(inp)
            correct += torch.sum(pred == label.data)

        test_loss /= len(loader.dataset)
        accuracy = float(correct) / float(len(loader.dataset))

        print('{} set: Average loss: {:.4f}, Accuracy: ({:.5f}%)'
            .format(mode, test_loss, accuracy))

        if save:
            return accuracy, preds, img_ids
        else:
            return accuracy     

# Copied from:　https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
def save_checkpoint(state, is_best, filename='./models/checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Saving new best model")
        shutil.copyfile(filename, './models/checkpoints/model_best.pth.tar')

# Copied from:　https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
def adjust_learning_rate(optimizer, epoch, lr, n_epochs, lr_decay):
    # Decay lr every n_epochs 
    lr = lr * (lr_decay ** (epoch // n_epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
     

################
## Parameters ##
################
train_file = args.train_file      
val_file = args.val_file
test_file = args.test_file    
data_root = args.data_root    
resume = args.checkpoint

img_size = args.size
n_classes = 1010
n_workers = args.n_workers
epochs = args.epochs
epochs_ft = args.epochs_ft
start_epoch = 0
batch_size = args.batch_size
lr = args.lr
lr_ft = args.lr_ft
lr_decay = args.lr_decay
momentum = args.momentum
weight_decay = args.weight_decay
n_epochs_lr = args.lr_decay_rate
n_epochs_lr_ft = args.lr_decay_rate_ft

ft_clip = args.ft_clip
print_freq = 100


###################
## Load datasets ##
###################
train_data = load_dataset.INAT(data_root, train_file, is_train=True, size=img_size)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

## Downsampled training data
# Load class weights from file
with open('./data/class_weights', 'rb') as f:
    class_weights = pickle.load(f)

class_weights = torch.DoubleTensor(class_weights)
sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(train_data) // ft_clip, replacement=True)
train_ft_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=n_workers, pin_memory=True)

val_data = load_dataset.INAT(data_root, val_file, is_train=False, size=img_size)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

test_data = load_dataset.INAT(data_root, test_file, is_train=False, size=img_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)


###################
## Prepare Model ##
###################
## Load pretrained inception3
model = inception_v3(pretrained=True)
model.fc = nn.Linear(2048, n_classes)
model.aux_logits = False
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

##############
## Training ##
##############
best_acc = 0.0

## Resume if checkpoint exists
# Copied from https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))


## Full Dataset Training Loop
start = time.time()
for epoch in range(start_epoch, epochs):
    # Decay lr every 2 epochs
    adjust_learning_rate(optimizer, epoch, lr, n_epochs_lr, lr_decay)

    # Train model for an epoch
    train(train_loader, model, criterion, optimizer, epoch, epochs)
    
    # Test on validation set
    acc = test(val_loader, model, criterion, save=False)

    # Save the best acc and current checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

end = time.time()-start
print('Training Completed in {:.0f}m {:.0f}s \n'.format(
        end // 60, end % 60))

if(start_epoch<epochs):
    start_epoch = epochs

## Downsampled Balanced Dataset
start = time.time()
for epoch in range(start_epoch, epochs+epochs_ft):
    # Decay lr every 4 epochs
    adjust_learning_rate(optimizer, epoch, lr_ft, n_epochs_lr_ft, lr_decay)

    # Train model for an epoch
    train(train_ft_loader, model, criterion, optimizer, epoch-epochs, epochs_ft)
    
    # Test on validation set
    acc = test(val_loader, model, criterion, save=False)

    # Save the best acc and current checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

end = time.time()-start
print('Fine-Tuning Completed in {:.0f}m {:.0f}s \n'.format(
        end // 60, end % 60))



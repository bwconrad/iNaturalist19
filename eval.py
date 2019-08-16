import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn

import numpy as np 
import pandas as pd 
import json 
import os
from PIL import Image 
import sys
import urllib.request
from torchsummary import summary
import time
import shutil
import argparse

import load_dataset
from models.inception3 import *

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./models/checkpoints/checkpoint.pth.tar')
parser.add_argument('--val_file', type=str, default='./data/val2019.json')
parser.add_argument('--test_file', type=str, default='./data/test2019.json')
parser.add_argument('--data_root', type=str, default='./data/')
parser.add_argument('--output_file', type=str, default='output.csv')
parser.add_argument('--size', type=int, default=560)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'])
args = parser.parse_args()


def test(loader, model, criterion, save=False, mode='Validation'):
    with torch.no_grad():
        model.eval() # Set model to testing mode
        
        test_loss = 0.0 
        correct = 0.0 

        preds = []
        img_ids = []

        print('Evaluating on {}'.format(mode))
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

def eval(loader, model):
    with torch.no_grad():
        model.eval() # Set model to testing mode

        preds = []
        img_ids = []

        print('Evaluating on Test Set')
        for inp, img_id, label in loader:
            inp = inp.to(device)

            # Make prediction with model
            output = model(inp)
            _, pred = torch.max(output, 1)
            
            # Save prediction
            img_ids.append(img_id.cpu().numpy().astype(np.int))
            preds.append(pred.cpu().numpy().astype(np.int))
        
        return preds, img_ids



# Parameters
mode = args.mode
val_file = args.val_file
test_file =  args.test_file
data_root = args.data_root     
output_file = args.output_file  
resume = args.weights

save_preds=True

img_size = args.size
n_classes = 1010
batch_size = args.batch_size
n_workers = 4


# Load data
if mode == 'val':
    val_data = load_dataset.INAT(data_root, val_file, is_train=False, size=img_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

else:
    test_data = load_dataset.INAT(data_root, test_file, is_train=False, size=img_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

# Load Model
model = inception_v3(pretrained=False)
model.fc = nn.Linear(2048, n_classes)
model.aux_logits = False
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)


# Load weights
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})"
        .format(resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(resume))
    sys.exit()



# Evaluate
print('Evaluating:')
if mode == 'val':
    acc, preds, img_ids = test(val_loader, model, criterion, save=True)
    print(acc)
else:
    preds, img_ids = eval(test_loader, model)

# Save predictions to csv
# Convert list of batch predictions into a single list
preds = [item for sublist in preds for item in sublist]
img_ids = [item for sublist in img_ids for item in sublist]

submission = pd.DataFrame({'id': img_ids, 'predicted': preds}).set_index('id')
submission.to_csv(output_file)
print('Submission File Created')


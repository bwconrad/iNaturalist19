# iNaturalist19
Training code for an Inception 3 model on the iNaturalist 2019 fine grain classification
dataset. 

After 30 training epochs and 46 fine-tuning the model achieved an accuracy of 72.3% on the
test set which was 47th in the [Kaggle
competition](https://www.kaggle.com/c/inaturalist-2019-fgvc6/leaderboard).  

Techniques used on the network are:
+ Data augmentation with the ImageNet AutoAugment policies
  ([Link](https://arxiv.org/abs/1805.09501))
+ Training on high resoluition 560x560 images 
  ([Link](http://openaccess.thecvf.com/content_cvpr_2018/html/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.html)) *
+ Fine-tuning on a balanced subset of the full dataset with a small learning rate
  ([Link](http://openaccess.thecvf.com/content_cvpr_2018/html/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.html))
  **


\* The large increase in number of flops from training on high resolution images did not give a
worth while advantage over training on a lower resolution for more epochs. 448x448  is likely
ideal for training on a single GPU.

\** The technique described in the paper is simply  a slightly fancier version of down sampling that gave marginal
performance increase. 
  
## Requirements

+ Python 3 
+ Pytorch

#### Required Files
+ Download the iNaturalist 2019 dataset from
  [here](https://github.com/visipedia/inat_comp).
+ Download the pretrained Inception 3 weights from
  [here](http://vision.caltech.edu/~macaodha/inat2018/iNat_2018_InceptionV3.pth.tar).
+ Organize the downloaded files into following file hierarchy:
```
iNaturalist19
│   ...
│
└───data
│   │   class_weights
│   │   test2019.json
│   │   train2019.json
│   │   val2019.json
│   │ 
│   └───test2019 
│   │   │  ...
│   │   │  
│   │
│   └───train_val2019
│       │  ...
│       │  
│            
└───models
│   │   iNat_2018_InceptionV3.pth.tar 
│   │   inception3.py
│   │            
│   └───checkpoints
│       │  *empty*         
│       │  
│
│   ...
│

```

## Usage
To train, run `python3 train.py` which will train the network for 10 epochs on the entire
dataset and 10 epochs on the balanced dataset with 1/4 the size.

To evaluate a  model, run `python eval.py` which will evaluate `checkpoint.pth.tar` on the
validation set. 

#### Arugments
+ `train.py`:
    + `--checkpoint`: file to resume training from (default: none)
    + `--train_file`: training set annotation file 
    + `--val_file`: validation set annotation file
    + `--test_file`: testing set anotation file
    + `--data_root`: directory where `test2019` and `train_val2019` directories are found
    + `--epochs`: number of epochs on full dataset
    + `--epochs_ft`: number of epochs for fine-tuning
    + `--size`: resolution of the images (default: 560)
    + `--batch_size`: (default: 16)
    + `--lr`: learning rate for full dataset training (default: 0.0075)
    + `--lr_ft`: learning rate for fine-tuning (default: 0.00075)
    + `--momentum`: (default: 0.9)
    + `--weight_decay`: (default: 1e-4)
    + `--lr_decay`: proportion for learning rate decay (default: 0.94) 
    + `--lr_decay_rate`: per how many epochs does the lr decay during training (default: 2)
    + `--lr_decay_rate_ft`: per how many epochs does the lr decay during fine-tuning (default: 4)
    + `--n_workers`: (default: 4)
    + `--ft_clip`: shrink the dataset to 1/ft_clip the size of the full dataset (default: 4)

+ `eval.py`:
    + `--mode`: 
        + `val`: evaluate on validation set (gives accuracy)
        + `test`: evaluate on test set (no accuracy)
    + `--weights`: checkpoint file of trained model
    + `--val_file`: validation set annotation file
    + `--test_file`: testing set anotation file
    + `--data_root`: directory where `test2019` and `train_val2019` directories are found
    + `--size`: resolution of the images (default: 560)
    + `--batch_size`: (default: 16)



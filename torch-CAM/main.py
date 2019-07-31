import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms

import data
import extract

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.manual_seed(1)
device = torch.device("cuda:0")
kwargs = {'num_workers':1, 'pin_memory':True}

image_transform_params = {'image_mode': 'shrink',  'output_image_size': {'width':224, 'height':224}}
target_transform_params = {'target_mode': 'only_cls', 'image_transform_params' : image_transform_params}

imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]) 

image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing]) 

train_dataset, val_dataset = data.make_trainval_dataset(
    image_transform_params = image_transform_params,
    transform = image_transform,
    target_transform_params = target_transform_params,
    download = True
)
print(train_dataset[0])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
           batch_size=32,
           shuffle=True,
           **kwargs)

valid_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        **kwargs)
############################################ Model
model = extract.FeatureExtractor()
model = model.to(device=device)

########################################### Preprocessing the dataset to extract the features
print("Extracting and saving the features for the training set")
extract.extract_save_features(train_loader, model, device, "./features_train/train_")

print("Extracting and saving the features for the validation set")
extract.extract_save_features(valid_loader, model, device, "./features_valid/valid_")

print("Done.")


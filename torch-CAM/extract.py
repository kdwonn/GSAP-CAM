import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os.path
import sys

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)

        # We freeze the networks
        # we will just perform forward passes, so we do not
        # need to compute any gradients
        # Does the following eliminate some computations ?
        for param in model.parameters():
            param.requires_grad = False

        # We keep only the feature maps of all the models
        # sometimes we can directly access a "features" attribute
        # sometimes we need to explictely extract the layers from the childrens
        self.body = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.body(x)

def extract_save_features(loader: torch.utils.data.DataLoader,
                          model: torch.nn.Module,
                          device: torch.device,
                          filename_prefix: str):

    with torch.no_grad():
        model.eval()
        batch_idx = 1
        for (inputs, targets) in tqdm(loader):

            inputs = inputs.to(device=device)

            # Compute the forward propagation through the body
            # just to extract the features
            features = model(inputs)
            torch.save({'features':features, 'targets':targets},
                       filename_prefix+"{}.pt".format(batch_idx))

            batch_idx += 1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPoweredPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, expotent=1):
        super(GlobalPoweredPool2d, self).__init__()
        self.expotent = expotent

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = x.view(-1, 2048, 49)
        powered = torch.pow(x, self.expotent)
        powered_sum = torch.add(torch.sum(powered, 2), 1e-10)

        weighted_sum = torch.sum(torch.mul(x, powered), 2)
        return torch.div(weighted_sum, powered_sum).view(-1, 2048)

class GlobalAveragePool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, expotent=1):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = x.view(-1, 2048, 49)
        sum = torch.sum(x, 2)
        return sum.view(-1, 2048)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.GPP = GlobalPoweredPool2d(expotent=1)
        self.fc = nn.Linear(2048, 20)
    def forward(self, input):
        output = self.GPP(input)
        output = self.fc(output)
        output = F.sigmoid(output)
        return output

class ClassifierBase(nn.Module):
    def __init__(self):
        super(ClassifierBase, self).__init__()
        self.GAP = GlobalAveragePool2d(expotent=1)
        self.fc = nn.Linear(2048, 20)
    def forward(self, input):
        output = self.GAP(input)
        output = self.fc(output)
        output = F.sigmoid(output)
        return output

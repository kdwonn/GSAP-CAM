import torch.nn.init as init
import torch.utils
import glob
from torchvision import datasets, transforms
from tqdm import tqdm


import data
import extract

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.manual_seed(1)
device = torch.device("cuda:0")
kwargs = {'num_workers':1, 'pin_memory':True}

tensor_filename = './train_1.pt'
data, target = torch.load(tensor_filename, map_location='cpu')

print(len(data))
print(len(target))

print(data[0])
print(len(data[0]))
print(len(data[0][0]))
print(len(data[0][0][0]))
print(data.size())
print(target)

x = data.view(32, 2048, -1)
powered = torch.pow(x, 1)
powered_sum = torch.add(torch.sum(powered, 2), 1e-10)
weighted_sum = torch.sum(torch.mul(x, powered), 2)
result = torch.div(weighted_sum, powered_sum).view(32, 2048)
print(result)
print(result.size())
print(result[0].size())
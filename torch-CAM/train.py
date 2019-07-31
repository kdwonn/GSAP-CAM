import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import glob
from torchvision import datasets, transforms
from tqdm import tqdm

import model as cls_model

import os

def load_tensors(path_regex):
    """
        Load a collection of tensors that match the regular expression path
        The individual tensors are loaded in CPU
        Returns a concatenated tensors of all the loaded tensors.
    """
    tensors = None
    for tensor_filename in tqdm(glob.glob(path_regex), ncols=50):
        tensor = torch.load(tensor_filename, map_location='cuda:0')
        if not tensors:
            tensors = tensor
        else:
            for k in tensors:
                tensors[k] = torch.cat((tensors[k], tensor[k]))
        del tensor
    for k in tensors:
        print("Key {} has shape {}".format(k, tensors[k].shape))
    return tensors

def train(loader, epoch, device, opt, model):
    model.train()
    loss_sum = 0.0
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        opt.zero_grad()
        prediction = model(data)
        loss = F.binary_cross_entropy(prediction, target)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        if False :
            print('{} \t Loss: {:.6f}'.format(i, loss.item()))
    print('\tTrain loss in epoch {}: {:7.5f}'.format(epoch, loss_sum / 179))

def test(loader, epoch, device, model):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            o = model(data)
            o_np = o.cpu().numpy()
            
            threshold = lambda x: 1 if (x >= 0.5) else -1
            vthreshold = np.vectorize(threshold)
            prediction = vthreshold(o_np).astype(int)

            target = label.cpu().numpy().astype(int)
            accuracy = np.sum(prediction == target) / (np.count_nonzero(target))

            loss_sum += F.binary_cross_entropy(o, label).item()
            acc_sum += accuracy
    print('\ttest in epoch {} avg. loss: {:7.5f}, acc: {:6.4f}'.format(epoch, loss_sum/len(loader), 100 * acc_sum/len(loader)))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.manual_seed(22)
    device = torch.device("cuda:0")
    kwargs = {'num_workers':0, 'pin_memory':False}

    train_data = load_tensors('./features_train/train_*.pt')
    valid_data = load_tensors('./features_valid/valid_*.pt')

    train_data_set = torch.utils.data.TensorDataset(train_data['features'], train_data['targets'])
    valid_data_set = torch.utils.data.TensorDataset(valid_data['features'], valid_data['targets'])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=32, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data_set, batch_size=32, shuffle=False, **kwargs)

    model = cls_model.Classifier()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    epoch = 30

    for i in range(epoch):
        train(train_loader, i, device, optimizer, model)
        test(valid_loader, i, device, model)

    torch.save(model, 'best_model.pt')

    # model = cls_model.ClassifierBase()
    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    # epoch = 30

    # for i in range(epoch):
    #     train(train_loader, i, device, optimizer, model)
    #     test(valid_loader, i, device, model)

    # torch.save(model, 'base_model.pt')

if __name__ == '__main__':
    main()
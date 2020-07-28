import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.distributions as distributions
import torch.utils.data as data
from tqdm import tqdm

import RealNVP, utils

def train(model, train_loader, val_loader, optimizer, max_epochs = 100, device = 'gpu'):
    epoch = 0
    train_losses = []
    val_losses = []
    while epoch < max_epochs:
        model.train()
        batch_history = []
        for _, (img, _) in enumerate(tqdm(train_loader)):
            logit_x, log_det = logit_transform(img)
            logit_x = logit_x.to(device)
            log_det = log_det.to(device)

            log_prob = model.log_prob(logit_x)
            log_prob += log_det

            # calculate loss for this batch
            batch_loss = - torch.mean(log_prob) / (3.0 * 32.0 * 32.0) 
            batch_history.append(float(batch_loss.data))

            # gradient updates
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()   

        epoch_loss = np.mean(batch_history)  
        tqdm.write(f'Epoch {epoch} Loss: {epoch_loss:.2f}')
        train_losses.append(epoch_loss)
        val_losses.append(get_loss(model, val_loader, device))
        epoch += 1
        if epoch % 5 == 0:
            torch.save(model, "model_realnvp_epoch_{}.model".format(str(epoch)))

    torch.save(model, "realnvp_final.model")
    return model

def get_loss(model, val_loader, device):
    model.eval()
    errors = []
    for _, (img, _) in enumerate(val_loader):
        logit_x, log_det = logit_transform(img)
        logit_x = logit_x.to(device)
        log_det = log_det.to(device)

        log_prob = model.log_prob(logit_x)
        log_prob += log_det

        loss = -torch.mean(log_prob) / (3.0 * 32.0 * 32.0)
        errors.append(float(loss.data))
    return np.mean(errors)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    [train_split, val_split] = data.random_split(trainset, [int(0.9 * len(trainset)), int(0.1 * len(trainset))])
    trainloader = torch.utils.data.DataLoader(train_split, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_split, batch_size=64, shuffle = True) 
    
    model = RealNVP()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model = train(model, trainloader, valloader, optimizer, max_epochs = 20, device = device)

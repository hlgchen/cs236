"""
In this file models are implemented for the method described in 
A. Voynov and A. Babenko, “Unsupervised discovery of interpretable directions in the gan latent
space,” in International Conference on Machine Learning, pp. 9786–9796, PMLR, 2020.

Parts of code is adopted from: https://github.com/anvoynov/GANLatentDiscovery/blob/master/latent_shift_predictor.py
In particular: 
- Reconstructor is LeNetShiftPredictor
- Reconstructor2 is LatentShiftPredictor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .default_model import *  


class A(nn.Module): 
    def __init__(self, z_dim, n_directions):
        super().__init__()

        self.k = n_directions
        self.z_dim = z_dim

        self.a = nn.Parameter(torch.rand((z_dim, n_directions)))

    def forward(self, z, labels): 
        onehot = F.one_hot(labels[0].long(), num_classes=self.k).float()
        return z + (onehot@self.a.transpose(0,1) * labels[1].unsqueeze(-1))


class ANorm(nn.Module): 
    def __init__(self, z_dim, n_directions):
        super().__init__()

        self.A = nn.utils.weight_norm(A(z_dim, n_directions), name='a')
        

    def forward(self, z, labels): 
        return self.A(z, labels)


class Reconstructor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(Reconstructor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.convnet(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()



def save_hook(module, input, output):
    setattr(module, 'output', output)


class Reconstructor2(nn.Module):
    def __init__(self, dim, downsample=None):
        super(Reconstructor2, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class Reconstructor3(nn.Module): 
    def __init__(self, discriminator, dim): 
        super(Reconstructor3, self).__init__()
        self.features_extractor = discriminator
        self.features_extractor = nn.Sequential(*list(self.features_extractor.children())[:-2])

        for param in self.features_extractor.parameters():
            param.requires_grad = False

        # self.convnet = nn.Sequential(
        #     nn.Conv2d(2* 1024, 2* 512, kernel_size=(1, 1)),
        #     nn.BatchNorm2d(2* 512),
        #     nn.ReLU(),
        #     nn.Conv2d(2* 512, 2* 256, kernel_size=(2, 2)),
        #     nn.BatchNorm2d(2* 256),
        #     nn.ReLU(),
        # )
        self.convnet = nn.Sequential(
            nn.Conv2d(2* 1024, 2* 256, kernel_size=(2, 2)),
            nn.BatchNorm2d(2* 256),
            nn.ReLU(),
        )
        
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1,x2 ): 
        x1 = self.features_extractor(x1)
        x2 = self.features_extractor(x2)

        x = self.convnet(torch.cat([x1, x2], dim=1)).squeeze()

        logits = self.type_estimator(x)
        shift = self.shift_estimator(x)

        return logits, shift




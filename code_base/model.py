import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *

"""
This module contains the models for InfoGAN as described in 
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, “Infogan: Interpretable
representation learning by information maximizing generative adversarial nets,”
"""


class Generator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        ld (dict): Dictionary with information about latent variables.
                expected keys: 
                    - nz: number of latent variables z
                    - n_dis_c: number of discrete c variables
                    - dis_c_dim: dimension of discrte c
                    - n_con_c: number of continuous c variables
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, ld, ngf=256, bottom_width=4):
        super().__init__()
        
        zc = int(ld['nz'] + ld['n_dis_c'] * ld['dis_c_dim'] + ld['n_con_c'])
        self.l1 = nn.Linear(zc, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf, upsample=True)
        self.block3 = GBlock(ngf, ngf, upsample=True)
        self.block4 = GBlock(ngf, ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        y = torch.tanh(h)
        return y


class DiscriminatorRaw32(nn.Module):
    r"""
    ResNet backbone raw-discriminator for SNGAN. 
    The output of the module is NOT the discriminator output. 
    The output subsequently has to be passed through Discriminator(32)
    to get the Discriminator output. 
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.activation = nn.ReLU(True)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        return h


class Generator64(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        ld (dict): Dictionary with information about latent variables.
                expected keys: 
                    - nz: number of latent variables z
                    - n_dis_c: number of discrete c variables
                    - dis_c_dim: dimension of discrte c
                    - n_con_c: number of continuous c variables
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, ld, ngf=1024, bottom_width=4):
        super().__init__()

        zc = int(ld['nz'] + ld['n_dis_c'] * ld['dis_c_dim'] + ld['n_con_c'])
        self.l1 = nn.Linear(zc, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y


class DiscriminatorRaw64(nn.Module):
    r"""
    ResNet backbone raw-discriminator for SNGAN. 
    The output of the module is NOT the discriminator output. 
    The output subsequently has to be passed through Discriminator(64)
    to get the Discriminator output. 
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.activation = nn.ReLU(True)


    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        return h


class Discriminator(nn.Module):
    """
    Discriminator(head). The input to this module is expected to be 
    the output of either DiscriminatorRaw32 or DiscriminatorRaw64.
    """
    def __init__(self, model=32, ndf=128): 
        super().__init__()
        if ndf is not None: 
            ndf = 128 if (model == 32) else 1024
        self.l = SNLinear(ndf, 1)
        nn.init.xavier_uniform_(self.l.weight.data, 1.0)
    
    def forward(self, h): 
        y = self.l(h)
        return y


class QHead(nn.Module):
    """This module can be stacked on top of DiscriminatorRaw32 or DiscriminatorRaw64. 
    If stacked the resulting model is the Q model for InfoGAN. 
    """
    def __init__(self, ld, model=32, ndf=None): 
        super().__init__()
        if ndf is None: 
            ndf = 128 if (model == 32) else 1024
        self.conv1 = DBlock(ndf, ndf, downsample = False)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.conv_disc = nn.Linear(ndf, ld['n_dis_c'] * ld['dis_c_dim'])
        self.conv_mu = nn.Linear(ndf, ld['n_con_c'])
        self.conv_var = nn.Linear(ndf, ld['n_con_c'])
    
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True).squeeze()

        disc_logits = self.conv_disc(x)

        mu = self.conv_mu(x)
        var = torch.exp(self.conv_var(x))

        return disc_logits, mu, var

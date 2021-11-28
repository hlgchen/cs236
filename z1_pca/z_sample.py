import sys 
import os
p = os.path.abspath('.')
if p not in sys.path:
    sys.path.append(p)

import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from default_model import *
import numpy as np
import pca_utils as pu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(n_img=16, direction=False, seed =None): 
    torch.manual_seed(seed)
    model = pu.load_model()
    
    if direction: 
        ls= []
        for i in range(6): 
            ls.append(pu.line_sample_z(n_img, 128))
        samples_z= torch.cat(ls)
    else:
        samples_z = torch.randn(n_img*6, 128)
    samples = model(samples_z)
    samples = F.interpolate(samples, 256).cpu()

    print("min:", samples.min())
    print("max:", samples.max())

    samples = vutils.make_grid(samples, nrow=n_img, padding=0, normalize=True)

    pu.get_figure(samples, filename=f'random_sample_seed{seed}')


def sample_direction(n_img=11, seed =None,  base=None): 

    model = pu.load_model()
    torch.manual_seed(seed)

    # create directions
    dir_ls = []
    dir_ls.append(torch.zeros(128))
    dir_ls[0][torch.randint(0,127,(1,))] = 1
    dir_ls.append(torch.zeros(128))
    dir_ls[1][torch.randint(0,127,(10,))] = 1
    dir_ls.append(torch.zeros(128))
    dir_ls[2][torch.randint(0,127,(50,))] = 1
    dir_ls.append(torch.zeros(128))
    dir_ls[3][torch.randint(0,127,(50,))] = torch.randn(50)
    dir_ls.append(torch.randn(1, 128))

    # create base image
    if base is None: 
        base = torch.randn(1, 128)

    line_s = pu.line_sample_z

    grid_ls = []
    for i in range(5):
        samples_z = line_s(base=base,
                            n=n_img, 
                            dim=128,  
                            dir=dir_ls[i],) 

        samples = model(samples_z)
        samples = F.interpolate(samples, 256).cpu()
        grid_ls.append(samples)
    
    samples = torch.cat(grid_ls)
    samples = vutils.make_grid(samples, nrow=n_img, padding=0, normalize=True)

    pu.get_figure(samples, filename=f'random_direction_seed{seed}_diameter_2', show=False)


def sample_pca(n_img=11, seed=None, base=None): 

    model = pu.load_model()
    torch.manual_seed(seed)

    U = torch.load('generated_images/eigenmatrix1000.pt')
    print("shape of eigenmatrix:",U.shape)

    if base is None: 
        base = torch.rand(1, 128)

    line_s = pu.line_sample_z

    grid_ls = []
    for i in range(5):
        samples_z = line_s(base=base,
                            n=n_img, 
                            dim=128,  
                            dir=U[i],) 

        samples = model(samples_z)
        samples = F.interpolate(samples, 256).cpu()
        grid_ls.append(samples)
    
    samples = torch.cat(grid_ls)
    samples = vutils.make_grid(samples, nrow=n_img, padding=0, normalize=True)

    pu.get_figure(samples, filename=f'pca_seed{seed}_setp_2', show=False)


def calculate_pca(seed = 825, pca_sample=1500): 
    model = pu.load_model()
    U = pu.get_eigenvectors(model, pca_sample, seed=seed)
    torch.save(U, f'generated_images/eigenmatrix_seed{seed}_sample{pca_sample}.pt')


def calculate_pca_inc(seed = 825, pca_sample=1500): 
    model = pu.load_model()
    U = pu.get_eigenvectors_incremental(model, pca_sample, seed=seed)
    torch.save(U, f'generated_images/eigenmatrix_inc_seed{seed}_sample{pca_sample}.pt')


def check_robustness_of_eigenvectors(pca, n=1000): 

    model = pu.load_model()
    U = torch.load(pca)

    U1 = pu.get_eigenvectors(model, n)
    U2 = pu.get_eigenvectors(model, n)

    cos = torch.nn.CosineSimilarity()

    print(f"cosinesimilarity with new PCA estimate {n}: {cos(U, U1)}")
    print(f"cosinesimilarity with new PCA estimate {n}: {cos(U, U2)}")


def apply_direction(model, base_ls, dir, dirname, diameter =1.5, n_img=11, seed=None):
    # torch.manual_seed(seed)
    grid_ls = []
    for i in range(4): 
        base = base_ls[i]
        samples_z = pu.line_sample_z(base=base,
                            n=n_img, 
                            dim=128,  
                            dir=dir,
                            diameter=diameter) 

        grid_ls.append(samples_z)

    # create more samples to stabilize predictions
    stb = 5
    for i in range(stb): 
        grid_ls.append(torch.randn(n_img, 128))
    samples_z = torch.cat(grid_ls)
    samples = model(samples_z)
    samples = F.interpolate(samples, 256).cpu()
    

    samples = vutils.make_grid(samples[:-stb * n_img], nrow=n_img, padding =0, 
                                normalize=True)

    pu.get_figure(samples, filename=f'{dirname}_samples_seed{seed}_diameter_{diameter}', show=False)


def direction_plots(seed = None, i=0):

    model = pu.load_model()
    
    U = torch.load('generated_images/eigenmatrix1000.pt')

    torch.manual_seed(seed)

    # create directions
    # dir_ls = []
    # dir_ls.append(torch.zeros(128))
    # dir_ls[0][torch.randint(0,127,(1,))] = 1
    # dir_ls.append(torch.zeros(128))
    # dir_ls[1][torch.randint(0,127,(10,))] = 1
    # dir_ls.append(torch.zeros(128))
    # dir_ls[2][torch.randint(0,127,(50,))] = 1
    # dir_ls.append(torch.zeros(128))
    # dir_ls[3][torch.randint(0,127,(50,))] = torch.randn(50)
    # dir_ls.append(torch.randn(1, 128))
    dir = torch.randn(1, 128)

    base_ls = torch.randn(4, 128)
    for diameter in [1.5, 3 ,5, 10, 20, 50]: 
        # apply_direction(model, base_ls, U[i],f"pca{i+1}", diameter =diameter, seed=seed)
        # apply_direction(model, base_ls, dir,f"random{i+1}", diameter =diameter, seed=seed)
        apply_direction(model, base_ls, -1* (U[1] + U[4]),f"pca2_5", diameter =diameter, seed=seed)


if __name__ == "__main__": 
    # sample(seed =255)
    # sample_direction(seed =35)
    # sample_pca(seed=25)
    # for i in range(3): 
    #     direction_plots(i=1)
    seed = torch.seed()
    seed = 17151426571300
    direction_plots(seed =seed, i=4)


    # calculate_pca(seed=1, pca_sample=150)
    # check_robustness_of_eigenvectors(n=150)

    # calculate_pca_inc(seed=1, pca_sample=2000)
    # check_robustness_of_eigenvectors("generated_images/eigenmatrix_inc_seed1_sample1000.pt", n=1000)

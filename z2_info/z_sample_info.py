import os 
from os.path import exists
import sys  

p = os.path.abspath('.')
if p not in sys.path:
    sys.path.append(p)

import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from model import *

import sys
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(ld, n=65000, res=64): 
    """Loads pretrained model with n steps and image resolution of res."""
    s = f"out/info4_{res}/ckpt/{n}.pth"
    state_dict = torch.load(s, map_location=device)
    if res==32: 
        model = Generator32(ld) 
    else: 
        model = Generator64(ld)
    model.load_state_dict(state_dict["net_g"])

    return model

def get_figure(samples,filename=None,show=True, auto_increase=True, label=True): 
    """Expects samples to have shape (3, height, width)"""
    fig, ax = plt.subplots(1,1, figsize = (10,8))
    plt.imshow(samples.permute(1, 2, 0), norm=None)

    if label: 
        ax.set_xticks([ax.get_xlim()[1]/2])
        ax.set_xticklabels(["Original Image"], fontsize= 20)
    else: 
        ax.set_xticks([])
    ax.set_yticks([])
    if filename is not None: 
        path = f"generated_images/info_gan/{filename}"
        if auto_increase: 
            while(exists(path + '.png')): 
                path = path + '1'
        fig.savefig(path + '.png', bbox_inches='tight')
    if show: 
        plt.show()

def noise_sample(latent_dimensions, batch_size, device=device):
    """
    Sample random noise vector for training.
    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of incompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """
    nz = latent_dimensions.get('nz', 0)
    n_dis_c = latent_dimensions.get('n_dis_c', 0)
    dis_c_dim = latent_dimensions.get('dis_c_dim', 0)
    n_con_c = latent_dimensions.get('n_con_c', 0) 


    z = torch.randn(batch_size, nz).to(device)

    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
        
        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        # con_c = torch.rand(batch_size, n_con_c, device=device) * 2 - 1
        con_c = torch.randn(batch_size, n_con_c, device=device)

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((noise, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise

def change_info_direction(sample_zc, idx, cont, latent_dimensions, diameter=6, n_img=7): 
    ls = []
    if cont: # continuous latent direction
        for i in range(-(n_img//2), (n_img//2) + 1): 
            p = sample_zc.clone()
            p[:, idx] = p[:, idx] + i * (diameter/n_img)
            ls.append(p)
    else: 
        for i in range(latent_dimensions["dis_c_dim"]): 
            p = sample_zc.clone()
            for j in range(latent_dimensions["dis_c_dim"]): 
                p[:, idx + j] = 0
            p[:, idx + i] = 1
            ls.append(p)
    return ls

def get_c_list(ld, n_comparissons): 
    ls = []
    for dis in range(ld['n_dis_c']):
        ls.append((ld['nz'] + dis * ld['dis_c_dim'], False))
    for cont in range(ld['n_con_c']): 
        ls.append((ld['nz'] + ld['n_dis_c'] * ld['dis_c_dim'] + cont, True))
    for comparisson in random.sample(range(ld['nz']), n_comparissons): 
        ls.append((comparisson, True))
    return ls


def sample_info(latent_dimensions, seed = 825, n_diff_pic=5, n_comparissons=5, n_img =7): 

    model = load_model(ld=latent_dimensions, n=65000, res=64).eval()

    print(latent_dimensions)
    c_ls = get_c_list(latent_dimensions, n_comparissons)

    for c in c_ls: 
        ls = []
        torch.manual_seed(seed)
        random.seed(seed)
        for j in range(n_diff_pic): 
            sample_zc = noise_sample(latent_dimensions, 1)
            ls = ls + change_info_direction(sample_zc, c[0],c[1], latent_dimensions , n_img=n_img)

        # create more samples to stabilize predictions
        torch.manual_seed(seed)
        stb = 5
        for i in range(stb): 
            ls.append(torch.randn(n_img, sample_zc.shape[1]))
        sample_zc = torch.cat(ls, dim=0)       

        samples = model(sample_zc).detach()
        # samples = F.interpolate(samples, 256).cpu()

        n_row = (sample_zc.size(0) - stb * n_img)//n_diff_pic
        samples = vutils.make_grid(samples[:-stb * n_img], nrow=n_row, padding=0, normalize=True, range=(-1,1))

        dimension_type = 'cont' if c[1] else 'dis'
        label_flag = True if c[1] else False
        get_figure(samples, 
                   filename=f'info_seed{seed}_dir{c[0]}_{dimension_type}_{c[0]>latent_dimensions["nz"]}',
                   show=False, 
                   label=label_flag)


if __name__ == "__main__":
    seed = torch.seed()
    latent_dimensions = {
        'nz': 128, 
        'n_dis_c':2,
        'dis_c_dim':5,
        'n_con_c':5,
    }
    sample_info(latent_dimensions, seed)
import numpy as np 
import torch 

from os.path import exists
from default_model import *
import sys
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def line_sample_z(n , dim, base=None, dir=None, diameter=1.5, seed=None): 
    """Samples n vectors of dimension dim. The vectors are all in a line."""

    if seed is not None: 
        torch.manual_seed(seed)

    if base is None: 
        base = torch.randn(1, dim)

    if dir is None: 
        dir = torch.randn(1, dim)

    if dir[dir.abs() == dir.abs().max()].mean() < 0: 
        dir *= -1
    dir = dir.reshape(1, dim)
    dir = torch.nn.functional.normalize(dir,p=2, dim =1) * diameter
    # dir /= dir.abs().max()

    samples = [base + i *(1/n) * dir for i in np.arange(-(n//2), (n//2) + 1, 1)]
    sample = torch.cat(samples)

    return sample


def get_eigenvectors(model, N, seed=None): 

    y, mu, samples_z = get_y_activations(model, N, seed=None)

    _, S, V = torch.pca_lowrank(y, center=True)

    x = (y-mu)@V
    
    U= torch.linalg.lstsq(x,samples_z).solution

    return U 

def get_eigenvectors_incremental(model, N, seed=None): 

    y, mu, samples_z = get_y_activations(model, N, seed=None)

    from estimators import IPCAEstimator
    est = IPCAEstimator(6)
    est.fit(y)

    V, stdev, var_ratio = est.get_components()
    V = torch.tensor(V, dtype = torch.float).transpose(0,1)

    x = (y-mu)@V
    
    U= torch.linalg.lstsq(x,samples_z).solution

    return U 

def get_y_activations(model, N, seed=None): 
    if seed is not None: 
        torch.manual_seed(seed)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.l1.register_forward_hook(get_activation('l1'))

    samples_z = torch.randn(N, 128)
    _ = model(samples_z)

    y = activation['l1']
    mu = y.mean(dim = 0)

    return y, mu, samples_z


def load_model(n=295000, res=64): 
    """Loads pretrained model with n steps and image resolution of res."""
    s = f"out/baseline-{res}-{n//1000}k/ckpt/{n}.pth"
    state_dict = torch.load(s, map_location=device)
    if res==32: 
        model = Generator32() 
    else: 
        model = Generator64()
    model.load_state_dict(state_dict["net_g"])

    return model


def get_figure(samples,filename=None,show=True, auto_increase=True): 
    """Expects samples to have shape (3, height, width)"""
    fig, ax = plt.subplots(1,1, figsize = (20,10))
    plt.imshow(samples.permute(1, 2, 0), norm=None)

    ax.set_xticks([ax.get_xlim()[1]/2])
    ax.set_xticklabels(["Original Image"], fontsize= 20)
    ax.set_yticks([])
    if filename is not None: 
        path = f"generated_images/{filename}"
        if auto_increase: 
            while(exists(path + '.png')): 
                path = path + '1'
        fig.savefig(path + '.png', bbox_inches='tight')
    if show: 
        plt.show()


if __name__ == "__main__": 
    pass
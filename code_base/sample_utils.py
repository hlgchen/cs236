import numpy as np 
import torch 

from os.path import exists
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def line_sample_z(n , dim=128, base=None, dir=None, diameter=6, seed=None): 
    """
    Returns n datapoints of dimension dim. 
    The datapoints are all on line along direction dir.

    Params: 
        - n: number of datapoints to retrun 
        - dim: dimension of each datapoint
        - base: anchor of line (starting datapoint)
        - dir: direction along which to find the other datapoints
        - diameter: max l2 length between datapoints returned 
                (go from base diameter/2 units in positive dir direction 
                and the same in negative direction)
        - seed: Integer random seed

    Return
        - sample: torch.Tensor of size (n, dim)
    """

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

    samples = [base + i *(1/n) * dir for i in np.arange(-(n//2), (n//2) + 1, 1)]
    sample = torch.cat(samples)

    return sample


def save_figure(samples,filepath=None, x_label=True, show=True, auto_increase=True): 
    """
    Creates images out of samples, saves them if filename provided.
    Params: 
        - samples: torch.Tensor with shape (3, height, width)
        - filepath: string with filepath from generated_images 
        - x_label: if True original image label is added to image
        - show: if True, image is showed 
        - auto_increase: If True, if file already exists image will be saved with 
                same name with additional "1" in the end.
    """
    samples = samples.detach()
    fig, ax = plt.subplots(1,1, figsize = (20,10))
    plt.imshow(samples.permute(1, 2, 0), norm=None)
    
    if x_label: 
        ax.set_xticks([ax.get_xlim()[1]/2])
        ax.set_xticklabels(["Original Image"], fontsize= 20)
    else: 
        ax.set_xticks([])
    ax.set_yticks([])
    if filepath is not None: 
        path = f"generated_images/{filepath}"
        if auto_increase: 
            while(exists(path + '.png')): 
                path = path + '1'
        fig.savefig(path + '.png', bbox_inches='tight')
    if show: 
        plt.show()


if __name__ == "__main__": 
    pass
import torch
import torchvision.utils as vutils

from code_base.default_model import *
import code_base.sample_utils as su
from code_base.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, n_img=16, seed =None): 
    torch.manual_seed(seed)

    samples_z = torch.randn(n_img*6, 128)
    samples = model(samples_z)
    samples = F.interpolate(samples, 256).cpu()

    print("min:", samples.min())
    print("max:", samples.max())

    samples = vutils.make_grid(samples, nrow=n_img, padding=0, normalize=True)

    su.save_figure(samples, filename=f'random/random_sample_seed{seed}')


def apply_direction(model, base_tensor, dir, output_path, diameter, n_row_img=5):

    grid_ls = []
    for i in range(base_tensor.size(0)): 
        base = base_tensor[i]
        samples_z = su.line_sample_z(base=base,
                            n=n_row_img, 
                            dim=128,  
                            dir=dir,
                            diameter=diameter) 

        grid_ls.append(samples_z)

    # create more samples to stabilize predictions
    stb = 5
    for i in range(stb): 
        grid_ls.append(torch.randn(n_row_img, 128))
    samples_z = torch.cat(grid_ls)
    samples = model(samples_z)
    samples = F.interpolate(samples, 256).cpu()
    

    samples = vutils.make_grid(samples[:-stb * n_row_img], nrow=n_row_img, padding =0, 
                                normalize=True)

    su.save_figure(samples, filepath=output_path, show=False)


def direction_plots(model, output_path, dir, diameter_ls=[1.5, 3.5], n_row_img=5, n_col_img =5, seed=None):
    """Creates plots for direction with diameter in diameter_ls."""

    torch.manual_seed(seed)
    base_tensor = torch.randn(n_col_img, 128)

    for diameter in diameter_ls: 
        output_path_file = f"{output_path}_diameter{diameter}"
        apply_direction(model=model, 
                        base_tensor=base_tensor, 
                        dir=dir,
                        output_path=output_path_file, 
                        diameter=diameter, 
                        n_row_img=n_row_img)


def generate_random_images(model): 
    # create random_plot
    dir = torch.randn(128)
    for seed in [17151426571300, 4582673360057174413, 15891558334906035504]: 
        random_out_path = f"random/{seed}_random"
        direction_plots(model=model, 
                        output_path=random_out_path,
                        dir=dir,
                        seed =seed, 
                        diameter_ls=[1.5, 3, 6])

def generate_pca_images(model): 
    # create pca plots
    U = torch.load(f'out/pca/principal_directions_sample1000.pt')
    for i in [1, 4]: 
        dir = U[i]
        for seed in [17151426571300, 4582673360057174413, 15891558334906035504]: 
            pca_out_path = f"pca/{seed}_pca{i}"
            direction_plots(model=model, 
                            output_path=pca_out_path,
                            dir=dir,
                            seed =seed, 
                            diameter_ls=[1.5, 3])

def generate_reconstruction_images(model): 
    # create rec plots
    A = torch.load(f'out/rec/resnet_a10_s1500.pt')["A"]["A.a_v"].transpose(0,1)
    for i in range(10):
        dir = A[i]
        for seed in [17151426571300, 4582673360057174413, 15891558334906035504]: 
            rec_out_path = f"rec/{seed}_a{i}"
            direction_plots(model=model, 
                            output_path=rec_out_path,
                            dir=dir,
                            seed =seed, 
                            diameter_ls=[6])


if __name__ == "__main__": 

    model = load_model("net_g")

    # generate_random_images(model)
    generate_pca_images(model)
    generate_reconstruction_images(model)





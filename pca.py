import torch 
import code_base.utils as utils


def get_y_activations(model, n, seed=None): 
    """"
    Return activations of first linear layer of default model. 

    Params: 
        - model: default genarator model (either Generator64 or Generator32)
                from code_base.default_model
        - n: number of latent variables to pass through the first layer,
                in other words: number of activations to get back.
        - seed: integer random seed
    """
    if seed is not None: 
        torch.manual_seed(seed)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.l1.register_forward_hook(get_activation('l1'))

    samples_z = torch.randn(n, 128)
    _ = model(samples_z)

    y = activation['l1']
    mu = y.mean(dim = 0)

    return y, mu, samples_z


def get_projected_eigenvectors(model, n, seed=None): 
    """
    Calculates 6 Eigenvectors of activations of first linear layer, 
    then projects these Eigenvectors back into latent space. 

    Params: 
        - model: default genarator model (either Generator64 or Generator32)
                from code_base.default_model
        - n: number of latent variables to pass through the first layer,
                in other words: number of activations to get back.

    Returns: 
        - torch.tensor with 6 directions
    """

    y, mu, samples_z = get_y_activations(model, n, seed=None)

    _, _, V = torch.pca_lowrank(y, center=True)

    x = (y-mu)@V
    
    U= torch.linalg.lstsq(x,samples_z).solution

    return U 

def calculate_pca(seed = 825, n=1500): 
    """Calculates and saves principal direction estimate. 
    
    Params: 
        - save_path: (string) folder where principal directions should be saved
        - seed: integer seed
        - n: number of samples to be used for PCA estimation
    """
    print("calculating PCA")
    model = utils.load_model()
    U = get_projected_eigenvectors(model, n, seed=seed)
    path_saved = f'out/pca/principal_directions_seed{seed}_sample{n}.pt'
    torch.save(U, path_saved)
    print(f"principal directions saved at {path_saved}")
    return U
    


def check_robustness_of_eigenvectors(pca_file, n=1000): 
    """Calculates and print cosine similarity of principal directions saved at 
    pca_file with 2 seperate estimates. 
    """

    print("checking robustness")

    model = utils.load_model()
    U = torch.load(f"out/pca/{pca_file}")

    U1 = get_projected_eigenvectors(model, n)
    U2 = get_projected_eigenvectors(model, n)

    cos = torch.nn.CosineSimilarity()

    print(f"cosinesimilarity with new PCA estimate {n}: {cos(U, U1)}")
    print(f"cosinesimilarity with new PCA estimate {n}: {cos(U, U2)}")


if __name__ == "__main__": 

    torch.manual_seed(12345)
    seed = 1
    n = 1000

    U = calculate_pca(seed=seed, n=n)
    check_robustness_of_eigenvectors(
        pca_file = f"principal_directions_seed{seed}_sample{n}.pt", n=n
    )
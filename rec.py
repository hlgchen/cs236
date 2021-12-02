"""
This file is used to train the A direction Matrix and the Reconstructor R 
as described in 
A. Voynov and A. Babenko, “Unsupervised discovery of interpretable directions in the gan latent
space,” in International Conference on Machine Learning, pp. 9786–9796, PMLR, 2020.
"""

import time

import torch
from torch.nn.functional import cross_entropy
from torch.nn.functional import l1_loss

from code_base.rec_models import *
from code_base.default_model import *  
from code_base.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_z(batch_size): 
    return torch.randn(batch_size, 128, device=device)

def sample_direction(batch_size, num_dir=10): 
    return torch.randint(0, num_dir, (batch_size,), device=device)

def sample_magnitude(batch_size, magnitude_range=3): 
    return (torch.rand(batch_size, device=device) * 2 - 1) * magnitude_range

def get_labels(batch_size, num_dir=10): 
    direction = sample_direction(batch_size, num_dir).unsqueeze(0)
    magnitude = sample_magnitude(batch_size).unsqueeze(0)

    return torch.cat([direction, magnitude], dim=0)

def generate_training_data(gen, A, batch_size, num_dir=10): 
    z1 = sample_z(batch_size)
    i1 = gen(z1)

    labels = get_labels(batch_size, num_dir)
    z2 = A(z1, labels)
    i2 = gen(z2)

    return (i1, i2), labels


def lossfn(pred, labels): 
    k_loss = cross_entropy(input=pred[0], target=labels[0].long())
    e_loss = l1_loss(input=pred[1].squeeze(), target=labels[1])

    return k_loss + 0.25 * e_loss


def train_model(model_name, num_dir=10,  cont=None, start=0): 
    start_time = time.time()
    z_dim = 128
    batch_size=64
    gen = load_model('net_g').to(device)
    

    if model_name == "lenet": 
        clf = Reconstructor(dim=num_dir).to(device)
    elif model_name == "resnet": 
        clf = Reconstructor2(dim = num_dir).to(device)
    else: 
        clf = Reconstructor3(load_model('net_d'), dim=num_dir).to(device)

    A = ANorm(z_dim, num_dir).to(device)

    params = [
        {'params': clf.parameters()},
        {'params': A.parameters()}
    ]
    adam = torch.optim.Adam(params)

    if cont is not None: 
        path = f"out/rec/{cont}"
        A.load_state_dict(torch.load(path)['A'])
        clf.load_state_dict(torch.load(path)['clf'])

        start = int(cont[:-3].split("_")[-1][1:])

    loss_ls = []
    for it in range(start+ 1, start+5000): 
        adam.zero_grad()

        X, labels = generate_training_data(gen, A, batch_size, num_dir)
        pred = clf(X[0], X[1])

        loss = lossfn(pred=pred, labels=labels)
        loss.backward()
        adam.step()
        loss_ls.append(loss.item())
        
        print(f"iteration {it} loss is {loss}")
        if (it % 100 == 0) & (it >0): 
            print("---File saved... ellapsed time:  %s seconds ---" % (time.time() - start_time))
            print(f"average_loss: {sum(loss_ls)/len(loss_ls)}")
            loss_ls = []
            state_dict = dict()
            state_dict['A'] = A.state_dict()
            state_dict['clf'] =clf.state_dict()
            path = f'out/rec/{model_name}_a{num_dir}_s{it}.pt'
            torch.save(state_dict, path)


if __name__ == "__main__": 
    # cont = "resnet_a10_s1500.pt"
    cont=None
    train_model("lenet", cont=cont)

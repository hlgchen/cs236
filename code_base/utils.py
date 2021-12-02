import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .default_model import Generator32 as DefaultGenerator32
from .default_model import Generator64 as DefaultGenerator64
from .default_model import Discriminator32 as DefaultDiscriminator32
from .default_model import Discriminator64 as DefaultDiscriminator64


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader


def load_model(n=295000, res=64 ): 
    """Loads pretrained model with n steps and image resolution of res."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s = f"out/baseline-{res}-{n//1000}k/ckpt/{n}.pth"
    state_dict = torch.load(s, map_location=device)
    if res==32: 
        model = DefaultGenerator32() 
    else: 
        model = DefaultGenerator64()
    model.load_state_dict(state_dict["net_g"])

    return model


def load_model(m='net_g', n=295000, res=64): 
    """Loads pretrained model with n steps and image resolution of res."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    s = f"out/baseline-{res}-{n//1000}k/ckpt/{n}.pth"
    state_dict = torch.load(s, map_location=device)
    if res==32: 
        if m == 'net_g': 
            model = DefaultGenerator32() 
        else: 
            model = DefaultDiscriminator32()
    else: 
        if m == 'net_g':
            model = DefaultGenerator64()
        else: 
            model = DefaultDiscriminator64()
    model.load_state_dict(state_dict[m])

    return model




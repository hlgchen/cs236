import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tbx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torchmetrics import IS, FID, KID


def prepare_data_for_inception(x, device):
    r"""
    Preprocess data to be feed into the Inception model.
    """

    x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
    minv, maxv = float(x.min()), float(x.max())
    x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
    x.mul_(255).add_(0.5).clamp_(0, 255)

    return x.to(device).to(torch.uint8)


def noise_sample(latent_dimensions, batch_size, device):
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


def prepare_data_for_gan(x, latent_dimensions, device):
    r"""
    Helper function to prepare inputs for model.
    """

    return (
        x.to(device),
        noise_sample(latent_dimensions, x.size(0), device),
    )


def compute_prob(logits):
    r"""
    Computes probability from model output.
    """

    return torch.sigmoid(logits).mean()


def hinge_loss_g(fake_preds):
    r"""
    Computes generator hinge loss.
    """

    return -fake_preds.mean()


def hinge_loss_d(real_preds, fake_preds):
    r"""
    Computes discriminator hinge loss.
    """

    return F.relu(1.0 - real_preds).mean() + F.relu(1.0 + fake_preds).mean()


def compute_loss_g(net_g, net_q, net_d_raw, net_dh, z, latent_dimensions, loss_func_g):
    r"""
    General implementation to compute generator loss.
    """

    fakes = net_g(z)
    fake_preds = net_dh(net_d_raw(fakes)).view(-1)
    loss_g_raw = loss_func_g(fake_preds)

    # mutual information
    nz = latent_dimensions.get('nz', 0)
    n_dis_c = latent_dimensions.get('n_dis_c', 0)
    dis_c_dim = latent_dimensions.get('dis_c_dim', 0)

    cat, mu, sigma = net_q(net_d_raw(fakes))

    cat_neg_log_prob = 0
    cat_loss = nn.CrossEntropyLoss()
    for j in range(n_dis_c): 
        cat_target = torch.argmax(
            z[:, (nz + j * dis_c_dim) : (nz + (j+1) * dis_c_dim)], dim=1)
        cat_neg_log_prob += cat_loss(cat[:, j*dis_c_dim: (j+1)*dis_c_dim], cat_target)

    normal = torch.distributions.normal.Normal(mu, sigma)
    con_neg_log_prob = - normal.log_prob(z[:, (nz + n_dis_c * dis_c_dim) : ]).sum(-1).mean() * 0.1  

    # add the losses 
    loss_g = loss_g_raw +  (cat_neg_log_prob + con_neg_log_prob)

    return loss_g, fakes, fake_preds, cat_neg_log_prob, con_neg_log_prob


def compute_loss_d(net_g, net_d_raw, net_dh, reals, z, loss_func_d):
    r"""
    General implementation to compute discriminator loss.
    """

    real_preds = net_dh(net_d_raw(reals)).view(-1)
    fakes = net_g(z).detach()
    fake_preds = net_dh(net_d_raw(fakes)).view(-1)
    loss_d = loss_func_d(real_preds, fake_preds)

    return loss_d, fakes, real_preds, fake_preds


def train_step(opt, sch, compute_loss):
    r"""
    General implementation to perform a training step.
    """

    loss = compute_loss()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sch.step()

    return loss


def evaluate(net_g, net_q, net_d_raw, net_dh, dataloader, latent_dimensions, device, samples_z=None):
    r"""
    Evaluates model and logs metrics.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        device (Device): Torch device to perform evaluation on.
        samples_z (Tensor): Noise tensor to generate samples.
    """

    net_g.to(device).eval()
    net_q.to(device).eval()
    net_d_raw.to(device).eval()
    net_dh.to(device).eval()

    with torch.no_grad():

        # Initialize metrics
        is_, fid, kid, loss_gs, loss_ds, real_preds, fake_preds = (
            IS().to(device),
            FID().to(device),
            KID().to(device),
            [],
            [],
            [],
            [],
        )
        cat_ls = []
        con_ls = []

        for data, _ in tqdm(dataloader, desc="Evaluating Model"):

            # Compute losses and save intermediate outputs
            reals, z = prepare_data_for_gan(data, latent_dimensions, device)
            loss_d, fakes, real_pred, fake_pred = compute_loss_d(
                net_g,
                net_d_raw,
                net_dh,
                reals,
                z,
                hinge_loss_d,
            )
            loss_g, _, _, cat_neg_log_prob,con_neg_log_prob = compute_loss_g(
                net_g,
                net_q,
                net_d_raw,
                net_dh,
                z,
                latent_dimensions,
                hinge_loss_g,
            )

            # Update metrics
            loss_gs.append(loss_g)
            loss_ds.append(loss_d)
            real_preds.append(compute_prob(real_pred))
            fake_preds.append(compute_prob(fake_pred))
            # reals = prepare_data_for_inception(reals, device)
            # fakes = prepare_data_for_inception(fakes, device)
            # is_.update(fakes)
            # fid.update(reals, real=True)
            # fid.update(fakes, real=False)
            # kid.update(reals, real=True)
            # kid.update(fakes, real=False)
            cat_ls.append(cat_neg_log_prob)
            con_ls.append(con_neg_log_prob)

        # Process metrics
        metrics = {
            "L(G)": torch.stack(loss_gs).mean().item(),
            "L(D)": torch.stack(loss_ds).mean().item(),
            "D(x)": torch.stack(real_preds).mean().item(),
            "D(G(z))": torch.stack(fake_preds).mean().item(),
            # "IS": is_.compute()[0].item(),
            # "FID": fid.compute().item(),
            # "KID": kid.compute()[0].item(),
            "cat_neg_log_prob": torch.stack(cat_ls).mean().item(),
            "con_neg_log_prob": torch.stack(con_ls).mean().item(),
        }

        # Create samples
        if samples_z is not None:
            samples = net_g(samples_z)
            samples = F.interpolate(samples, 256).cpu()
            samples = vutils.make_grid(samples, nrow=6, padding=4, normalize=True)

    return metrics if samples_z is None else (metrics, samples)


class Trainer:
    r"""
    Trainer performs GAN training, checkpointing and logging.
    Attributes:
        net_g (Module): Torch generator model.
        net_d (Module): Torch discriminator model.
        opt_g (Optimizer): Torch optimizer for generator.
        opt_d (Optimizer): Torch optimizer for discriminator.
        sch_g (Scheduler): Torch lr scheduler for generator.
        sch_d (Scheduler): Torch lr scheduler for discriminator.
        train_dataloader (Dataloader): Torch training set dataloader.
        eval_dataloader (Dataloader): Torch evaluation set dataloader.
        nz (int): Generator input / noise dimension.
        log_dir (str): Path to store log outputs.
        ckpt_dir (str): Path to store and load checkpoints.
        device (Device): Torch device to perform training on.
    """

    def __init__(
        self,
        net_g,
        net_q, 
        net_d_raw,
        net_dh,
        opt_g,
        opt_d,
        sch_g,
        sch_d,
        train_dataloader,
        eval_dataloader,
        latent_dimensions,
        log_dir,
        ckpt_dir,
        device,
    ):
        # Setup models, dataloader, optimizers
        self.net_g = net_g.to(device)
        self.net_q = net_q.to(device)
        self.net_d_raw = net_d_raw.to(device)
        self.net_dh = net_dh.to(device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sch_g = sch_g
        self.sch_d = sch_d
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training parameters
        self.device = device
        self.latent_dimensions = latent_dimensions
        self.step = 0

        # Setup checkpointing, evaluation and logging
        self.fixed_z = noise_sample(latent_dimensions, 32, device)
        self.logger = tbx.SummaryWriter(log_dir)
        self.ckpt_dir = ckpt_dir

    def _state_dict(self):
        return {
            "net_g": self.net_g.state_dict(),
            "net_q": self.net_q.state_dict(),
            "net_d_raw": self.net_d_raw.state_dict(),
            "net_dh": self.net_dh.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sch_g": self.sch_g.state_dict(),
            "sch_d": self.sch_d.state_dict(),
            "step": self.step,
        }

    def _load_state_dict(self, state_dict):
        self.net_g.load_state_dict(state_dict["net_g"])
        self.net_q.load_state_dict(state_dict["net_q"])
        self.net_d_raw.load_state_dict(state_dict["net_d_raw"])
        self.net_dh.load_state_dict(state_dict["net_dh"])
        self.opt_g.load_state_dict(state_dict["opt_g"])
        self.opt_d.load_state_dict(state_dict["opt_d"])
        self.sch_g.load_state_dict(state_dict["sch_g"])
        self.sch_d.load_state_dict(state_dict["sch_d"])
        self.step = state_dict["step"]

    def _load_checkpoint(self):
        r"""
        Finds the last checkpoint in ckpt_dir and load states.
        """

        ckpt_paths = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        if ckpt_paths:  # Train from scratch if no checkpoints were found
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            self._load_state_dict(torch.load(ckpt_path))

    def _save_checkpoint(self):
        r"""
        Saves model, optimizer and trainer states.
        """

        ckpt_path = os.path.join(self.ckpt_dir, f"{self.step}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def _log(self, metrics, samples):
        r"""
        Logs metrics and samples to Tensorboard.
        """

        for k, v in metrics.items():
            self.logger.add_scalar(k, v, self.step)
        self.logger.add_image("Samples", samples, self.step)
        self.logger.flush()

    def _train_step_g(self, z):
        r"""
        Performs a generator training step.
        """
        self.net_g.train()
        self.net_q.train()

        return train_step(
            self.opt_g,
            self.sch_g,
            lambda: compute_loss_g(
                self.net_g,
                self.net_q, 
                self.net_d_raw,
                self.net_dh,
                z,
                self.latent_dimensions,
                hinge_loss_g,
            )[0],
        )

    def _train_step_d(self, reals, z):
        r"""
        Performs a discriminator training step.
        """
        self.net_d_raw.train()
        self.net_dh.train()

        return train_step(
            self.opt_d,
            self.sch_d,
            lambda: compute_loss_d(
                self.net_g,
                self.net_d_raw,
                self.net_dh,
                reals,
                z,
                hinge_loss_d,
            )[0],
        )

    def train(self, max_steps, repeat_d, eval_every, ckpt_every):
        r"""
        Performs GAN training, checkpointing and logging.
        Attributes:
            max_steps (int): Number of steps before stopping.
            repeat_d (int): Number of discriminator updates before a generator update.
            eval_every (int): Number of steps before logging to Tensorboard.
            ckpt_every (int): Number of steps before checkpointing models.
        """

        self._load_checkpoint()

        while True:
            pbar = tqdm(self.train_dataloader)
            for data, _ in pbar:

                # Training step
                reals, z = prepare_data_for_gan(data, self.latent_dimensions, self.device)
                loss_d = self._train_step_d(reals, z)
                if self.step % repeat_d == 0:
                    loss_g = self._train_step_g(z)

                pbar.set_description(
                    f"L(G):{loss_g.item():.2f}|L(D):{loss_d.item():.2f}|{self.step}/{max_steps}"
                )

                if self.step != 0 and self.step % eval_every == 0:
                    self._log(
                        *evaluate(
                            self.net_g,
                            self.net_q,
                            self.net_d_raw,
                            self.net_dh,
                            self.eval_dataloader,
                            self.latent_dimensions,
                            self.device,
                            samples_z=self.fixed_z,
                        )
                    )

                if self.step != 0 and self.step % ckpt_every == 0:
                    self._save_checkpoint()

                self.step += 1
                if self.step > max_steps:
                    return

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(-1, self.shape, 1)        # shape: batch * feature * 1

class VASC_pytorch:
    def __init__(self, in_dim, latent = 2, var=False):
        self.in_dim = in_dim # expr.shape[1]
        self.latent = latent
        self.var = var

        self.dropout = nn.Dropout(0.5)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Linear(512, 128),
            nn.ReLU(True),          # inplace = True, in order to save memory
            nn.Linear(128, 32),
            nn.ReLU(True)
        )

        self.fc_mu = nn.Linear(32, latent)
        self.fc_var = nn.Linear(32, latent)

        self.decoder = nn.Sequential(
            nn.Linear(latent, 32),
            nn.ReLU(True),
            nn.Linear(32, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, in_dim),
            nn.Sigmoid()
        )

        # dropout rate p = exp(-x**2), need logp and log(1 - p) to calculate softmax
        self.dropout_rate = nn.Sequential(
            Lambda(lambda x: -x ** 2),
            Lambda(lambda x: torch.exp(x)),
            Lambda(lambda x: torch.log(x + 1e-20)),
            Reshape(self.in_dim)
        )
        self.nondropout_rate = nn.Sequential(
            Lambda(lambda x: -x ** 2),
            Lambda(lambda x: torch.exp(x)),
            Lambda(lambda x: 1 - x),
            Lambda(lambda x: torch.log(x + 1e-20)),
            Reshape(self.in_dim)
        )

    def loss_function(self, x, z_mean, z_log_var, x_decoded_mean):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons_loss = self.in_dim * F.binary_cross_entropy(x, x_decoded_mean)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1), dim=0)
        return {'loss': (recons_loss + kl_loss), 'Reconstruction_Loss':recons_loss, 'KLD':-kl_loss}

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # Split the result into mu and var components of the latent Gaussian distribution
        z_mean = self.fc_mu(result)
        if self.var:
            z_log_var = F.softplus(self.fc_var(result))
            return z_mean, z_log_var
        else:
            return z_mean

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, args):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param z_mean: (Tensor) Mean of the latent Gaussian [B x D]
        :param z_log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z_mean, z_log_var = args
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def scDropout_immitation(self, expr):
        expr_x_drop = self.dropout_rate(expr)                           # shape: batch * feature * 1
        expr_x_nondrop = self.nondropout_rate(expr)                     # shape: batch * feature * 1
        logits = np.concatenate((expr_x_drop, expr_x_nondrop), axis=-1) # shape: batch * feature * 2
        return logits

    def ZeroInflatedLayer(self, args, eps=1e-8):
        logits, tau = args
        uniform = torch.rand(logits.shape)                    # u ~ U(0, 1)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)  # g = -log (-log u)
        # may be wrong !
        gumbel_softmax = nn.Softmax((logits + gumbel) / tau)  # softamx: exp[(log p + g) / tau], exp[(log (1 - p) + g) / tau]
        samples = gumbel_softmax[:, :, 1]                     # select last column
        samples = samples.view(-1, self.in_dim)               # shape: batch * feature
        return samples

    def forward(self, args):
        expr, tau = args[0], args[1]
        expr = self.dropout(expr)
        z_log_var = torch.eye(self.latent)
        if self.var:
            z_mean, z_log_var = self.encode(expr)
        else:
            z_mean = self.encode(expr)
        z = self.reparameterize([z_mean, z_log_var])
        expr = self.decode(z)
        logits = self.scDropout_immitation(expr)
        tau = tau.repeat(1, 2).view(-1, 2, self.in_dim).transpose(1, 2)  # shape: batch * feature * 2
        out = torch.mul(expr, self.ZeroInflatedLayer([logits, tau]))
        return [expr, z_mean, z_log_var, out]

        # example(2 samples, 3 features)
        # tau = tensor([[1, 2, 5],
        #               [3, 4, 6]])]
        # tau.repeat(1, 2).view(-1, 2, 3).transpose(1, 2)
        # tensor([[[1, 1],
        #          [2, 2],
        #          [5, 5]],
        #         [[3, 3],
        #          [4, 4],
        #          [6, 6]]])
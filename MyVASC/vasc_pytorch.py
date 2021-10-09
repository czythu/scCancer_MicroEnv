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


class VascPytorch(nn.Module):
    def __init__(self, in_dim, latent=2, dist='Normal', gpu=False, var=False):
        super(VascPytorch, self).__init__()
        self.in_dim = in_dim # expr.shape[1]
        self.latent = latent
        self.dist = dist
        self.var = var
        self.gpu = gpu

        self.dropout = nn.Dropout(0.5)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Linear(512, 128),
            nn.ReLU(True),          # inplace = True, in order to save memory
            nn.Linear(128, 32),
            nn.ReLU(True)
        )

        self.fc_parm1 = nn.Linear(32, latent)
        self.fc_parm2 = nn.Linear(32, latent)

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

    # Compute the VAE loss function(latent distribution: Normal)
    def lossfunction_normal(self, x, z_mean, z_log_var, x_decoded_mean, kld_weight=1):
        recons_loss = self.in_dim * F.binary_cross_entropy_with_logits(x_decoded_mean, x) # for biase
        # recons_loss = F.mse_loss(x_decoded_mean, x)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1), dim=0)
        return {'loss': (recons_loss + kld_weight * kl_loss), 'Reconstruction_Loss': recons_loss, 'KLD': -kl_loss}

    # Compute the VAE loss function(latent distribution: Negative binomial)
    # z_mean, z_log_var = p * r / (1 - p), p * r / (1 - p) ** 2
    def lossfunction_nb(self, x, z_mean, z_log_var, x_decoded_mean, kld_weight=1):
        recons_loss = self.in_dim * F.binary_cross_entropy_with_logits(x_decoded_mean, x) # for biase
        # recons_loss = F.mse_loss(x_decoded_mean, x)
        # Error !
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1), dim=0)
        return {'loss': (recons_loss + kld_weight * kl_loss), 'Reconstruction_Loss': recons_loss, 'KLD': -kl_loss}

    def encode(self, input_matrix):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input_matrix: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_matrix)
        # Split the result into mu and var components of the latent Gaussian distribution
        if self.dist == 'Normal':
            z_mean = self.fc_parm1(result)
            if self.var:
                z_log_var = F.softplus(self.fc_parm2(result))
                return z_mean, z_log_var
            else:
                return z_mean
        else:
            # r∈R, p∈[0, 1]
            # r = torch.tensor(F.softplus(self.fc_parm1(result)).ceil(), dtype=torch.int64)
            r = F.softplus(self.fc_parm1(result))
            p = F.sigmoid(self.fc_parm2(result))
            return r, p

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

    def reparameterize_nb(self, r, p):
        nb_model = torch.distributions.negative_binomial.NegativeBinomial(total_count=r, probs=p)
        sample = nb_model.sample(sample_shape=torch.Size([1]))
        output = F.softplus(sample[0])                                          # F.sigmoid(sample[0])
        return output

    def reparameterize_normal(self, z_mean, z_log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param z_mean: (Tensor) Mean of the latent Gaussian [B x D]
        :param z_log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        if self.gpu:
            std, eps = std.cuda(), eps.cuda()
        # np.random.multivariate_normal(z_mean, )
        return z_mean + eps * std

    def sc_dropout_imitation(self, expr):
        expr_x_drop = self.dropout_rate(expr)                           # shape: batch * feature * 1
        expr_x_nondrop = self.nondropout_rate(expr)                     # shape: batch * feature * 1
        logits = torch.cat([expr_x_drop, expr_x_nondrop], dim=-1)       # shape: batch * feature * 2
        return logits

    def zero_inflated_layer(self, args, eps=1e-8):
        logits, tau = args
        uniform = torch.rand(logits.shape)                    # u ~ U(0, 1)
        if self.gpu:
            uniform = uniform.cuda()
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)  # g = -log (-log u)
        gumbel_softmax = F.softmax((logits + gumbel) / tau)   # exp[(log p + g) / tau], exp[(log (1 - p) + g) / tau]
        samples = gumbel_softmax[:, :, 1]                     # select last column
        samples = samples.view(-1, self.in_dim)               # shape: batch * feature
        return samples

    def forward(self, args):
        expr_ori, tau = args[0], args[1]
        expr = self.dropout(expr_ori)

        if self.dist == 'Normal':           # self.dist == 'Normal'
            z_log_var = torch.ones(expr.shape[0], self.latent)
            # z_log_var = torch.as_tensor(np.array([np.identity(self.latent) for i in range(expr.shape[0])]))
            if self.gpu:
                z_log_var = z_log_var.cuda()
            if self.var:
                z_mean, z_log_var = self.encode(expr)
            else:
                z_mean = self.encode(expr)
            z = self.reparameterize_normal(z_mean, z_log_var)
        else:                               # self.dist == 'Negative binomial'
            r, p = self.encode(expr)
            z_mean, z_log_var = p * r / (1 - p), p * r / (1 - p) ** 2
            z = self.reparameterize_nb(r, p)

        expr_new = self.decode(z)
        logits = self.sc_dropout_imitation(expr_new)
        tau = tau.repeat(1, 2).view(-1, 2, self.in_dim).transpose(1, 2)  # shape: batch * feature * 2
        out = torch.mul(expr, self.zero_inflated_layer([logits, tau]))
        return [expr_ori, z, z_mean, z_log_var, out]

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

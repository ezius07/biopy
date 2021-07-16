import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..utils import ReverseLayerF


def reparameterization(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


class ConvEncoder(nn.Module):
    def __init__(self, data_size=(3, 400, 400), latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim

        self.conv = nn.Sequential(                                  # 3x400x400
            nn.Conv2d(in_channel, 64, kernel_size=10, stride=4, padding=4),  # 64x100x100
            nn.LeakyReLU(.05, inplace=True),
            nn.MaxPool2d(kernel_size=6, stride=2),                  # 64x48x48
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # 192x48x48
            nn.LeakyReLU(.05, inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),                  # 192x23x23
            nn.BatchNorm2d(192),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=2, stride=2),          # 384x23x23
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(384),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # 256x23x23
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # 256x23x23
            nn.LeakyReLU(.05, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear = nn.Sequential(  # 256*6*6
            nn.Linear(256 * 6 * 6, 512),  # 512
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(512, 512),  # 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(.2, inplace=True),
        )

        self.mu = nn.Linear(512, self.latent_dim)
        self.logvar = nn.Linear(512, self.latent_dim)

        self.mu_tensor = None
        self.logvar_tensor = None

    def forward(self, img):
        x = self.conv(img)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        self.mu_tensor = self.mu(x)
        self.logvar_tensor = self.logvar(x)

        z = reparameterization(self.mu_tensor, self.logvar_tensor)

        return z


class ConvDecoder(nn.Module):

    def __init__(self, output_size=(3, 400, 400), latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_size = output_size

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512),  # 512
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(512, 512),  # 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(512, 256 * 6 * 6),  # 256*6*6
        )

        self.conv = nn.Sequential(                                   # 256x6x6
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),   # 128x14x14
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1),   # 128x14x14
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),    # 64x32x32
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=3, padding=2),     # 32x98x98
            nn.LeakyReLU(.05, inplace=True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, out_channels, kernel_size=10, stride=4, padding=1),      # 3x400x400
        )

    def forward(self, z):
        z = self.linear(z)
        z = z.view(z.shape[0], 256, 6, 6)
        img = self.conv(z)

        return img


#####################################  AAE  ######################################################


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class AAEImg(nn.Module):
    ae_type = 'AAE'
    def __init__(self, data_size=(3, 400, 400), hidden_size=2):
        super().__init__()

        self.encoder = ConvEncoder(data_size, hidden_size)
        self.decoder = ConvDecoder(data_size, hidden_size)

        self.discriminator = Discriminator(hidden_size)

    def forward(self, x, input_is_z=False, alpha=None):
        if not input_is_z:
            z = self.encoder(x)

            if alpha is not None:
                x = ReverseLayerF.apply(z, alpha)
                x = self.discriminator(x)
                return x

            x = self.decoder(z)

            return x, self.encoder.mu, self.encoder.logvar

        x = self.discriminator(x)

        return x

    def encode_and_sample(self, x):
        z = self.encoder(x)
        
        return z


############################## VAE ######################

class VEncoder(nn.Module):
    
    def __init__(self, backbone, trainable_blocks, pretrained=True, hidden_size=16):
        
        super().__init__()
        
        bkbn = backbone(pretrained=pretrained)
        self.end_bkbn_ch = bkbn.fc.in_features
        
        enc_layers = list(bkbn.children())[:-1]
        
        self.not_trainable_enc = nn.Sequential(*(enc_layers[:-trainable_blocks]))
        self.trainable_enc = nn.Sequential(*(enc_layers[-trainable_blocks:]))
        
        self.__freezeBackbone()
        
        self.enc = nn.Sequential(                                   
            self.not_trainable_enc,
            self.trainable_enc
        )
        
        self.enc_mean = nn.Linear(in_features=self.end_bkbn_ch, out_features=hidden_size, bias=True)
        self.enc_logvar = nn.Linear(in_features=self.end_bkbn_ch, out_features=hidden_size, bias=True)
    
    def __freezeBackbone(self):
        for param in self.not_trainable_enc.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        
        x = self.enc(x)
        x = x.view(-1, self.end_bkbn_ch)
        mean = self.enc_mean(x)
        log_var = self.enc_logvar(x)
        
        return mean, log_var


class ImgVAE(nn.Module):
    ae_type='VAE'
    def __init__(self, backbone, trainable_blocks, pretrained=True, hidden_size=16, output_size=(3, 400, 400)):
        
        super().__init__()
    
        self.encoder = VEncoder(backbone, trainable_blocks, pretrained=pretrained, hidden_size=hidden_size)
        self.decoder = ConvDecoder(latent_dim=hidden_size, output_size=output_size)
        
    def forward(self, x, return_z=False):
        
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        x = self.decoder(z)
        
        if return_z:
            return x, mean, log_var, z
            
        return x, mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z  
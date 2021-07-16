import torch
from torch import nn


class DomainTranslator(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z

    def decode(self, x):
        x = self.decoder(x)
        
        return x
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)

        x = self.decoder(z)
        
        return x

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        # [TODO]: Is it better to use randn_like or normal_ as in the original code?
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
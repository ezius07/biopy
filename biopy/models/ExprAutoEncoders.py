import torch
from torch import nn

from ..utils import ReverseLayerF

#########################  Encoder and Decoder for VAE/AE/SAAE ##################
class VEncoder(nn.Module):
    def __init__(self, input_size=58276, hidden_size=16):        
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_size, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512))
        self.enc_mean = nn.Linear(512, hidden_size)
        self.enc_logvar = nn.Linear(512, hidden_size)
        
    def forward(self, x):
        x = self.enc(x)
        mean = self.enc_mean(x)
        log_var = self.enc_logvar(x)
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_size=58276, hidden_size=16):        
        super().__init__()
        
        self.dec = nn.Sequential(nn.Linear(hidden_size, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, output_size))           
    
    
    def forward(self, x):
        x = self.dec(x)
        
        return x

###############################################################

#########################  VAE #####################

class VAE(nn.Module):
    ae_type='VAE'
    def __init__(self, data_size=131, hidden_size=16, **kwargs):
        super().__init__()
        
        self.encoder = VEncoder(input_size=data_size, hidden_size=hidden_size)
        self.decoder = Decoder(output_size=data_size, hidden_size=hidden_size)
        
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        x = self.decoder(z)
        return x, z, mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z

#########################  AAE #####################

class DoubleDiscriminator(nn.Module):
    """Latent space discriminator"""
    def __init__(self, hidden_size, n_out=2, n_hidden=80, **kwargs):
        super().__init__()
        self.nz = hidden_size
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(hidden_size, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden//2),
            nn.Linear(n_hidden//2, n_out)
        )

    def forward(self, x, alpha=None):
        if alpha is not None:
            x = ReverseLayerF.apply(x, alpha)
        x = self.net(x)
                
        return x

class Discriminator(nn.Module):
    def __init__(self, hidden_size, n_out=2, **kwargs):
        super().__init__()
        n_middle = max(hidden_size // 3, 3)
        self.net = nn.Sequential(nn.Linear(hidden_size, n_middle),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_middle, n_out))

    def forward(self, x, alpha=None, **kwargs):
        if alpha is not None:
                x = ReverseLayerF.apply(x, alpha)
        x = self.net(x)
                
        return x
        

class AAE(nn.Module):
    ae_type='AAE'
    def __init__(self, data_size=58276, hidden_size=16, num_distrib=2, **kwargs):
        super().__init__()
        
        self.encoder = VEncoder(input_size=data_size, hidden_size=hidden_size)
        self.decoder = Decoder(output_size=data_size, hidden_size=hidden_size)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 10),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(10, num_distrib))
        
    def forward(self, x, input_is_z=False, alpha=None):
        if not input_is_z:
            mean, log_var = self.encoder(x)
            z = self.sample(mean, log_var)

            if alpha is not None:
                x = ReverseLayerF.apply(z, alpha)
                x = self.discriminator(x)

                return x
            else:
                x = self.decoder(z)

            return x, z, mean, log_var
        else:
            x = self.discriminator(x)
            
            return x
        
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z


#########################  SAAE #######################

class ClassDiscriminatorBig(nn.Module):
    """Latent space discriminator"""
    def __init__(self, hidden_size, n_hidden=100, n_out=2, num_classes=5, repeat_labels_ntimes=20, **kwargs):
        super().__init__()
        self.nz = hidden_size
        self.repeat_labels_ntimes = repeat_labels_ntimes
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(hidden_size+num_classes*repeat_labels_ntimes, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, n_hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden//2, n_out)
        )

    def forward(self, x, alpha=None, labels=None):
        if alpha is not None:
            x = ReverseLayerF.apply(x, alpha)
   
        one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes)
        one_hot = one_hot.repeat_interleave(self.repeat_labels_ntimes, 1)
        x = torch.cat((one_hot, x), 1)
        x = self.net(x)
                
        return x


class ClassDiscriminator(nn.Module):
    def __init__(self, hidden_size, n_out=2, num_classes=5, **kwargs):
        super().__init__()
        n_middle = max(hidden_size // 3, 3)
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Linear(hidden_size+num_classes, n_middle),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_middle, n_out))

    def forward(self, x, alpha=None, labels=None):
        if alpha is not None:
            x = ReverseLayerF.apply(x, alpha)
   
        one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes)
        x = torch.cat((one_hot, x), 1)
        x = self.net(x)
                
        return x


class SupervisedAAE(nn.Module):
    ae_type='SAAE'
    def __init__(self, data_size=131, hidden_size=16, num_distrib=2, num_classes=5, discriminator=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = VEncoder(input_size=data_size, hidden_size=hidden_size)
        self.decoder = Decoder(output_size=data_size, hidden_size=hidden_size)
        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = nn.Sequential(nn.Linear(hidden_size+num_classes, 20),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(20, num_distrib))
        
    def forward(self, x, input_is_z=False, alpha=None, labels=None):
        if not input_is_z:
            mean, log_var = self.encoder(x)
            z = self.sample(mean, log_var)

            if alpha is not None:
                x = ReverseLayerF.apply(z, alpha)
                
                one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes)
                x = torch.cat((one_hot, x), 1)
                x = self.discriminator(x)

                return x
            else:
                x = self.decoder(z)

            return x, z, mean, log_var
        else:
            one_hot = nn.functional.one_hot(labels-1, num_classes=self.num_classes)
            x = torch.cat((one_hot, x), 1)
            x = self.discriminator(x)
            
            return x
        
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z



#########################  Encoder for plain AE #####################

class Encoder(nn.Module):
    def __init__(self, input_size=58276, hidden_size=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_size, 1024),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(512),
                                 nn.Linear(512, hidden_size))

    def forward(self, x):
        x = self.enc(x)
        return x


#########################  Plain AE #####################

class AE(nn.Module):
    def __init__(self, data_size=131, hidden_size=16):
        super().__init__()
        
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size)
        self.decoder = Decoder(output_size=data_size, hidden_size=hidden_size)
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z, 0, 0

    def encode_and_sample(self, x):

        return self.encoder(x)

    
######################## small Enc/Dec and AAE ###################


class VEncoder_small(nn.Module):
    def __init__(self, input_size=131, hidden_size=16):        
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_size, 50),
                                nn.ReLU(),
                                nn.BatchNorm1d(50))
        self.enc_mean = nn.Linear(50, hidden_size)
        self.enc_logvar = nn.Linear(50, hidden_size)
        
    def forward(self, x):
        x = self.enc(x)
        mean = self.enc_mean(x)
        log_var = self.enc_logvar(x)
        
        return mean, log_var

        
class Decoder_small(nn.Module):
    def __init__(self, output_size=131, hidden_size=16):        
        super().__init__()
        
        self.dec = nn.Sequential(nn.Linear(hidden_size, 50),
                                nn.ReLU(),
                                #nn.BatchNorm1d(50),
                                nn.Linear(50, output_size),
                                #nn.Sigmoid()
                                #nn.ReLU(),
                                #nn.BatchNorm1d(output_shape)
                                )           
    
    
    def forward(self, x):
        x = self.dec(x)
        
        return x


class AAE_small(nn.Module):
    ae_type='AAE'
    def __init__(self, data_size=131, hidden_size=16, num_distrib=2):
        super().__init__()
        
        self.encoder = VEncoder_small(input_size=data_size, hidden_size=hidden_size)
        self.decoder = Decoder_small(output_size=data_size, hidden_size=hidden_size)
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, 10),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(10, num_distrib))
        
    def forward(self, x, input_is_z=False, alpha=None):
        if not input_is_z:
            mean, log_var = self.encoder(x)
            z = self.sample(mean, log_var)

            if alpha is not None:
                x = ReverseLayerF.apply(z, alpha)
                x = self.discriminator(x)

                return x
            else:
                x = self.decoder(z)

            return x, mean, log_var
        else:
            x = self.discriminator(x)
            
            return x
        
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def encode_and_sample(self, x):
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        
        return z
import torch
from torch import nn

from ..utils import ReverseLayerF

def reparameterization(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)

def upscale(scale_factor=2, mode='nearest_neighbor_and_conv3x3', **kwargs):

    def nearest_neighbor_and_conv3x3():
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(kwargs['inplanes'], kwargs['planes'], kernel_size=3, stride=1, padding=1, bias=False))

    def nearest_neighbor_and_conv1x1():
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(kwargs['inplanes'], kwargs['planes'], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(kwargs['planes']),
            )

    return locals()[mode]()

def downscale(scale_factor=.5, mode='conv1x1', **kwargs):

    def conv1x1():
        stride = int(1 / scale_factor)
        return nn.Sequential(nn.Conv2d(kwargs['inplanes'], kwargs['planes'], kernel_size=1, stride=stride, bias=False),
                             nn.BatchNorm2d(kwargs['planes']))

    def conv3x3():
        stride = int(1 / scale_factor)
        return nn.Conv2d(kwargs['inplanes'], kwargs['planes'], kernel_size=3, stride=stride, padding=1, bias=False)

    return locals()[mode]()
    
class BasicBlockUpDownScale(nn.Module):
 
    def __init__(self, inplanes, planes, scale_factor=1):
        super().__init__()

        if scale_factor <= 0:
            raise ValueError('Scaling factor must be >= 0')
        
        self.residual_rescaler = None
        if scale_factor < 1:
            self.conv1 = downscale(scale_factor=scale_factor, mode='conv3x3', inplanes=inplanes, planes=planes)
            self.residual_rescaler = downscale(scale_factor=scale_factor, mode='conv1x1', inplanes=inplanes, planes=planes)
        elif scale_factor == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = upscale(scale_factor=scale_factor, mode='nearest_neighbor_and_conv3x3', inplanes=inplanes, planes=planes)
            self.residual_rescaler = upscale(scale_factor=scale_factor, mode='nearest_neighbor_and_conv1x1', inplanes=inplanes, planes=planes)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.residual_rescaler is not None:
            residual = self.residual_rescaler(x)
 
        out = out + residual
        out = self.relu(out)
 
        return out

def make_residuals_layer(inplanes, planes, n_blocks=2, scale_factor=1):
    blocks = []
    
    # The first basic block of this layer will actually perform the down/up scale
    blocks.append(BasicBlockUpDownScale(inplanes, planes, scale_factor=scale_factor))

    for _ in range(1, n_blocks):
        blocks.append(BasicBlockUpDownScale(planes, planes))

    return nn.Sequential(*blocks)

class ConvEncoder(nn.Module):

    def __init__(self, data_size=(3, 400, 400), latent_dim=2):

        super().__init__()
        
        self.latent_dim = latent_dim

        self.conv_in = nn.Sequential(
            nn.Conv2d(data_size[0], 10, kernel_size=8, stride=4, padding=2), # 10x100x100
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 48, kernel_size=6, stride=2, padding=0, bias=False), # 48x48x48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        ) 

        self.resnet = nn.Sequential(
            make_residuals_layer(inplanes=48, planes=96, n_blocks=1, scale_factor=.5), # 96x24x24
            make_residuals_layer(inplanes=96, planes=96, n_blocks=2, scale_factor=.5), # 96x12x12
            make_residuals_layer(inplanes=96, planes=96, n_blocks=3, scale_factor=.5), # 96x6x6
            #nn.Conv2d(96, 256, kernel_size=6, stride=1, padding=0, bias=False),
        )

        self.linear = nn.Sequential(                                # 256*6*6
            nn.Linear(96*6*6, 512),                                # 512
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.mu = nn.Linear(512, self.latent_dim)
        self.logvar = nn.Linear(512, self.latent_dim)
        
        self.mu_tensor = None
        self.logvar_tensor = None


    def forward(self, img):
        
        x = self.conv_in(img)
        x = self.resnet(x)

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
            nn.Linear(latent_dim, 512),                               
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),            
            nn.Linear(512, 96*6*6),
        )

        self.resnet = nn.Sequential(
            make_residuals_layer(inplanes=96, planes=96, n_blocks=3, scale_factor=2), # 96x12x12
            make_residuals_layer(inplanes=96, planes=96, n_blocks=2, scale_factor=2), # 96x24x24
            make_residuals_layer(inplanes=96, planes=96, n_blocks=1, scale_factor=2), # 96x48x48
            make_residuals_layer(inplanes=96, planes=16, n_blocks=1, scale_factor=2), # 16x96x96
        )

        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, output_size[0], kernel_size=10, stride=2),
        )

    def forward(self, z):
        
        z = self.linear(z)
        z = z.view(z.shape[0], 96, 6, 6)
        
        z = self.resnet(z)
        img = self.conv_out(z)
        
        return img


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

##############################################################################################  

class AAEImgResnet(nn.Module): 
    ae_type='AAE'
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
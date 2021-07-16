import torch

from torch import nn
from torch.autograd import Variable

from ..utils import ReverseLayerF

class Simple_Classifier(nn.Module):
    """Latent space discriminator"""

    def __init__(self, nz, n_out=2):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_out),
        )

    def forward(self, x):
        return self.net(x)

class Adversarial_Classifier(nn.Module):
    """Latent space discriminator"""

    def __init__(self, nz, n_out=2):
        super(Adversarial_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_out),
        )

    def forward(self, x, alpha=None):
        if alpha is None:
            return self.net(x)
        
        return self.net(ReverseLayerF.apply(x, alpha))

class Adversarial_ClassifierA549(nn.Module):
    """Latent space discriminator"""

    def __init__(self, nz=50, n_out=3):
        super(Adversarial_ClassifierA549, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, n_out)
        )

    def forward(self, x, alpha=None):
        if alpha is None:
            return self.net(x)
        
        return self.net(ReverseLayerF.apply(x, alpha))


class Encoder(nn.Module):

    def __init__(self, data_size, hidden_size, n_hidden=1024, last_hidden=None):
        super(Encoder, self).__init__()
        if last_hidden is None:
            self.last_hidden = n_hidden
        else:
            self.last_hidden = last_hidden
        self.nz = hidden_size
        self.n_input = data_size
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(self.n_input, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, self.last_hidden),
                                     nn.BatchNorm1d(self.last_hidden),
                                     nn.ReLU(inplace=True)
                                     )

        self.fc1 = nn.Linear(self.last_hidden, self.nz)
        self.fc2 = nn.Linear(self.last_hidden, self.nz)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)


class Decoder(nn.Module):
    def __init__(self, data_size, hidden_size, n_hidden=1024, last_hidden=None):
        super(Decoder, self).__init__()
        if last_hidden is None:
            self.last_hidden = n_hidden
        else:
            self.last_hidden = last_hidden
        self.nz = hidden_size
        self.n_input = data_size
        self.n_hidden = n_hidden
        self.decoder = nn.Sequential(nn.Linear(self.nz, self.last_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(self.last_hidden),
                                     nn.Linear(self.last_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, self.n_input)
                                     )

    def forward(self, z):
        return self.decoder(z)

#self.simple_classifier = Simple_Classifier(self.nz, n_out=n_out)

class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""

    ae_type = "VAE"

    def __init__(self, data_size, hidden_size, n_hidden=1024, last_hidden=None, **kwargs):
        super(FC_VAE, self).__init__()
        if last_hidden is None:
            self.last_hidden = n_hidden
        else:
            self.last_hidden = last_hidden
        self.nz = hidden_size
        self.n_input = data_size
        self.n_hidden = n_hidden

        self.encoder = Encoder(data_size, hidden_size, n_hidden, self.last_hidden)
        self.decoder = Decoder(data_size, hidden_size, n_hidden, self.last_hidden)

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparametrize(mu, logvar)

        res = self.decoder(z)

        return res, z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if std.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def encode_and_sample(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return z

    def generate(self, z):
        res = self.decoder(z)
        return res

class FC_SAAE(FC_VAE):
    ae_type = 'SAAE'
    def __init__(self, data_size, hidden_size, n_hidden=1024, last_hidden=None, **kwargs):
        super().__init__(data_size, hidden_size, n_hidden=n_hidden, last_hidden=last_hidden, **kwargs)


class ImageClassifier(nn.Module):
    def __init__(self, latent_variable_size, pretrained, nout=2):
        super(ImageClassifier, self).__init__()
        self.latent_variable_size = latent_variable_size
        self.feature_extractor = pretrained
        self.classifier = nn.Linear(latent_variable_size, nout)

    def forward(self, x):
        x, _ = self.feature_extractor.encode(x)
        x = self.classifier(x.view(-1, self.latent_variable_size))
        return x


class ImgEncoder(nn.Module):
    def __init__(self,  nc, ndf, latent_variable_size=128, batchnorm=False):
        super(ImgEncoder, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm

        self.encoder = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )
        self.bn_mean = nn.BatchNorm1d(latent_variable_size)
        self.fc1 = nn.Linear(ndf*8*2*2, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*2*2, latent_variable_size)


    def forward(self, x, **kwargs):
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*2*2)
        if self.batchnorm:
            return self.bn_mean(self.fc1(h)), self.fc2(h)
        else:
            return self.fc1(h), self.fc2(h)


class ImgDecoder(nn.Module):
    def __init__(self,  nc, ngf, latent_variable_size=128):
        super(ImgDecoder, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.latent_variable_size = latent_variable_size

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,  ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 64 x 64
        )
        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, ngf*8*2*2),
            nn.ReLU(inplace=True),
            )

    def forward(self, z, **kwargs):
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 2, 2)
        return self.decoder(h)


class NucleiImgVAE(nn.Module):
    def __init__(self, nc=1, ngf=128, ndf=128, latent_variable_size=128, imsize=64, batchnorm=False, *args, **kwargs):
        super().__init__()
 
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.imsize = imsize
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm

        # encoder
        self.encoder = ImgEncoder(nc, ndf, latent_variable_size, batchnorm)
        # decoder
        self.decoder = ImgDecoder(nc, ngf, latent_variable_size)
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
 
    def encode_and_sample(self, x):
        mu, logvar = self.encoder(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decoder(z)
        return res
 
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        res = self.decoder(z)
        return res, z, mu, logvar


class FC_Autoencoder(nn.Module):
    """Autoencoder"""
    def __init__(self, n_input, nz, n_hidden=512):
        super(FC_Autoencoder, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, nz),
                                )

        self.decoder = nn.Sequential(nn.Linear(nz, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding
        

class FC_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden=1024, n_out=2):
        super(FC_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
#            nn.Linear(n_hidden, n_hidden),
#            nn.ReLU(inplace=True),
 #           nn.Linear(n_hidden, n_hidden),
 #           nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden,n_out)
        )

    def forward(self, x):
        return self.net(x)
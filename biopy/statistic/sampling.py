import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ..utils import sample_encodings
from ..utils.plots import plot_2d, plot_3d
from ..statistic.distributions import available_distr


def sample_from_empirical_distribution(n, encodings, labels=None, get_labels=False):
    rand_rows = torch.randperm(encodings.shape[0])[:n]
    random_sample = encodings[rand_rows, :]
    if get_labels:
        labels = labels[rand_rows]
        return random_sample, labels
    return random_sample, None


class DistributionSampler():
    
    available_distr = available_distr
    
    def __init__(self):
        self.is_empirical = False
    
    def get_default_names(self):
        return list(self.available_distr.keys())
    
    def flower_latent(self, dim, **kwargs):
                
        mean = kwargs['mean']

        if 'cov_matrix' in kwargs.keys():
            cov_matrix = kwargs['cov_matrix']
        else:
            cov_matrix = None
            var = kwargs['var']

        means = torch.cat((-mean*torch.eye(dim),mean*torch.eye(dim)))

        if cov_matrix != None:
            covariances = cov_matrix.unsqueeze(0)
            for i in range(2*dim-1):
                cov_matrix = torch.roll(torch.roll(covariances[-1],1,1),1,0)
                covariances = torch.cat((covariances, cov_matrix.unsqueeze(0)))

        else:
            covariances = var.unsqueeze(0)
            for i in range(2*dim-1):
                cov_matrix = torch.roll(covariances[-1],1,0)
                covariances = torch.cat((covariances, cov_matrix.unsqueeze(0)))
                
        return means, covariances
    
    def name_parser(self, name, **kwargs):
        
        if name in available_distr.keys():
            means = self.available_distr[name]['means']
            covariances = self.available_distr[name]['covariances']
        
        else:
            fields = name.split('_')
            if fields[0] == 'flower':
                means, covariances = self.flower_latent(int(fields[1][:-1]), **kwargs)
            else:
                raise Exception("Not implemented yet.")
       
        return means, covariances
    
    def build_default(self, name, **kwargs):
        
        means, covariances = self.name_parser(name, **kwargs)
                
        n_distr = means.shape[0]

        if means.shape == covariances.shape:
            self.comp = D.Independent(D.Normal(means, torch.sqrt(covariances)), 1)
        else:
            self.comp = D.MultivariateNormal(means, covariances)
        self.mix = D.Categorical(torch.ones(n_distr,))
        #self.comp = D.MultivariateNormal(self.available_distr[name]['means'], self.available_distr[name]['covariances'])
        self.gmm = MixtureSameFamily(self.mix, self.comp)
        
    def plot_sampling(self, n=5000, points=None, red_to_dim=2):
        if red_to_dim > 3:
            print("You cannot ask to plot more than 3 dimensions")
            return
        m1 = self.gmm.sample((n,))
        dim = m1.shape[-1]
        if dim == 2:
            plt.subplots(figsize=(10,8))
            plt.scatter(m1[:,0].numpy(), m1[:,1].numpy())
            if points is not None:
                plt.scatter(points[:,0], points[:,1], c='red')
            lims = (-12, 12)
            plt.ylim(lims)
            plt.xlim(lims)
            plt.grid()
            plt.show()
        elif dim == 3:
            plot_3d(m1[:,0].numpy(), m1[:,1].numpy(), m1[:,2].numpy())
        else:
            tsned = TSNE(n_components=red_to_dim).fit_transform(m1)
            if red_to_dim == 2:
                plot_2d(tsned[:,0], tsned[:,1])
            elif red_to_dim == 3:
                plot_3d(tsned[:,0], tsned[:,1], tsned[:,2])
        
    def build_gaussian_mixture(self, means, covariances):
        n = means.shape[0]
        self.mix = D.Categorical(torch.ones(n,))
        if means.shape == covariances.shape:
            self.comp = D.Independent(D.Normal(means, torch.sqrt(covariances)), 1)
        else:
            self.comp = D.MultivariateNormal(means, covariances)
        self.gmm = MixtureSameFamily(self.mix, self.comp)

    def build_laplacian_mixture(self, locs, scales):
        n = locs.shape[0]
        self.mix = D.Categorical(torch.ones(n,))
        self.comp = D.Independent(D.laplace.Laplace(locs, scales), 1)
        self.gmm = MixtureSameFamily(self.mix, self.comp)

    def sample(self, shape, get_labels=False):
        # e.g. shape  = (64,)
        if self.is_empirical:
            sample, labels = sample_from_empirical_distribution(shape, self.encodings, 
                                                                labels=self.dataset.labels,
                                                                get_labels=get_labels)
        else:
            sample = self.gmm.sample((shape,))
            if get_labels:
                labels = torch.argmax(self.comp.log_prob(sample.unsqueeze(1)), dim=1) + 1
        if get_labels:
            return sample, labels
        return sample
    
    def build_empirical_from_latent(self, model, dataset):
        self.is_empirical = True
        self.dataset = dataset
        self.encodings = torch.Tensor(sample_encodings(model, dataset, tsne=False))
    
    def empirical_sample(self):
        batch_size = 64
        sample, labels = sample_from_empirical_distribution(batch_size, self.encodings, 
                                                            labels=dataset.labels)
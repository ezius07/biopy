import torch
from torch.utils import data
from torch.autograd import Function
import numpy as np
from sklearn.manifold import TSNE

from torch.utils.data import Sampler
import random

def sample_encodings(model, dataset, tsne=True, dim=3):
    '''Encode from dataset and optionally reduce dimensionality'''
    mini_batch = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = data.DataLoader(dataset, batch_size=mini_batch, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    model.eval()
    model.to(device)
    encodings = []
    for batch_features, _ in loader:
        batch_features = batch_features.to(device)

        # compute reconstructions
        z = model.encode_and_sample(batch_features)
        encodings.append(z.detach().cpu())

    encodings = torch.cat(encodings, dim=0)
    hidden_size = encodings.shape[-1]
    encodings_py = encodings.numpy().reshape(-1, hidden_size)
    if tsne:
        encodings_t = TSNE(n_components=dim).fit_transform(encodings_py)
        return encodings_t
    return encodings_py


def generate_model_optimizer(model_class, losses, parameters):
    returns = []
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model and load it to the specified device
    hidden_size = parameters['HIDDEN_SIZE']
    input_size = parameters['INPUT_SIZE']
    model = model_class(data_size=input_size, hidden_size=hidden_size).to(device)

    # create an optimizer object
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=parameters['LR'], 
                               momentum=parameters['MOMENTUM'], weight_decay=parameters['WEIGHT_DECAY'])
   
    # scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=parameters['STEP_SIZE'], 
                                          gamma=parameters['GAMMA'])
    
    returns.append(model)
    returns.append(optimizer)
    returns.append(scheduler)
    try:
        losses_built = []
        for loss in losses:
            losses_built.append(loss())
        returns += losses_built
    except TypeError:
        losses_built = losses()
        returns.append(losses_built)
      
    return tuple(returns)

    
class StepClass():
    def __init__(self, start=1, step_size=2, unit_step=2):
        self.value = start
        self.unit_step = unit_step
        self.step_size = step_size
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step % self.step_size == 0:
            self.value += self.unit_step


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None    

class PredictableRandomSampler(Sampler):
    def __init__(self, n, seed=0):
        self.n = n
        self.seed_generator = random.Random(seed)

    def __iter__(self):
        current_seed = self.seed_generator.randint(0, 2**32 - 1)
        indexes = list(range(self.n))
        random.Random(current_seed).shuffle(indexes)
        return iter(indexes)

    def __len__(self):
        return self.n
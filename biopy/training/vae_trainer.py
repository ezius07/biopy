import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from .trainer import Trainer
from ..utils import StepClass


class VAETrainer(Trainer):
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader)
        self.train_loss = None
        self.test_loss = None
        self.reconstruct_loss = nn.MSELoss()      
   
    def train_model(self, **kwargs):
        print("[|||] STARTED VAE TRAINING [|||]")
        cudnn.benchmark
        self.train_loss = ([], [])
        self.test_loss = []
        C = StepClass(unit_step=5, step_size=1)
        beta = 0.001
        for epoch in range(self.parameters['NUM_EPOCHS']):
            loss = 0
            mse = 0
            kld = 0
            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()

                # compute reconstructions
                outputs, _, mean, log_var = self.model(batch_features)

                # compute loss
                mse_loss = self.reconstruct_loss(outputs, batch_features)
                kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                step_loss = mse_loss + max(beta*(kld_loss - C.value), 0)

                # compute accumulated gradients
                step_loss.backward()
                self.optimizer.step()


                # add the mini-batch training loss to epoch loss
                loss += step_loss.item()
                mse += mse_loss.item()
                kld += kld_loss.item()
                #print("mse =", mse_loss.item())
            # compute the epoch training loss
            loss = loss / len(self.train_loader)
            mse = mse / len(self.train_loader)
            kld = kld / len(self.train_loader)
            # display the epoch training loss
            if (epoch + 1) % self.parameters['LOG_FREQUENCY'] == 0:
                print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, self.parameters['NUM_EPOCHS'], loss))
            test = self.validate_on(self.model, self.test_loader, self.reconstruct_loss, self.device, is_vae=True)

            self.train_loss[0].append(mse)
            self.train_loss[1].append(kld)
            self.test_loss.append(test)
            self.scheduler.step()
            C.step()
        
  
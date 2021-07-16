import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from .trainer import Trainer
from ..utils import StepClass
from ..models.ExprAutoEncoders import Discriminator


class AAETrainer(Trainer):
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader, **kwargs)
        self.train_loss = None
        self.test_loss = None
        self.metric_train = None
        self.metric_test = None
        self.reconstruct_loss = nn.MSELoss()   
        self.discriminate_loss = nn.CrossEntropyLoss()
    
    def generate_models_optimizers(self, model_class, discriminator_class=Discriminator, optimizer_SGD=False, **kwargs):
        super().generate_models_optimizers(model_class=model_class, optimizer_SGD=optimizer_SGD, **kwargs)
        
        self.discriminator = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=2).to(self.device)

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.parameters.get('LR_D', self.parameters['LR']), 
                                        weight_decay=self.parameters.get('WEIGHT_DECAY_D', self.parameters['WEIGHT_DECAY']))

    def train_model(self, sampler=None, **kwargs):
        if sampler is None:
            return Exception("You should specify a sampler")
        print("[|||] STARTED AAE TRAINING [|||]")
        cudnn.benchmark
        # --- Log losses ---
        self.train_loss = ([], [])
        self.test_loss = []

        # --- Collect parameters ---
        alfa = self.parameters.get('ALPHA', 2)
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 60)
        BATCH_SIZE = self.parameters.get('BATCH_SIZE', 32)
        metric_classes = kwargs.get('metric', 0)
        save_models = self.parameters.get('SAVE_MODELS', False)
        eval_freq = kwargs.get('eval_freq', 1)

        # --- Set up discriminator labels ---
        z_dist_label = torch.zeros(BATCH_SIZE).long().to(self.device)
        true_dist_label = torch.ones(BATCH_SIZE).long().to(self.device)

        # --- Set up metrics ---
        self.setup_metrics(metric_classes)
        
        # --- Set up model saving ---
        self.setup_model_saving(save_models)

        for epoch in range(NUM_EPOCHS):
            avg_epoch_loss = 0
            avg_epoch_loss_discr = 0

            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                self.discriminator.train()
                self.optimizer_discriminator.zero_grad()

                # compute reconstructions
                outputs, z, mean, log_var = self.model(batch_features)

                # real batch through discriminator
                discr_z_dist = self.discriminator(z, alpha=1)

                # sampled batch through discriminator
                other_z = sampler.sample(BATCH_SIZE).to(self.device)
                discr_true_distr = self.discriminator(other_z, alpha=1)
 
                # compute loss
                mse_loss = self.reconstruct_loss(outputs, batch_features)
                z_dist_cse_loss = self.discriminate_loss(discr_z_dist, z_dist_label)
                true_dist_cse_loss = self.discriminate_loss(discr_true_distr, true_dist_label)

                loss = mse_loss + alfa*(z_dist_cse_loss + true_dist_cse_loss)

                # compute accumulated gradients
                loss.backward()
                self.optimizer.step()
                self.optimizer_discriminator.step()

                # add the mini-batch training loss to epoch loss
                avg_epoch_loss += mse_loss.item()
                avg_epoch_loss_discr += z_dist_cse_loss.item() + true_dist_cse_loss.item()

            # compute the epoch training loss
            avg_epoch_loss /=  len(self.train_loader)
            avg_epoch_loss_discr /= 2*len(self.train_loader)
            
            # display the epoch training loss
            if (epoch + 1) % self.parameters['LOG_FREQUENCY'] == 0:
                print("epoch : {}/{}, loss = ({:.4f}, {:.4f}),  LR = {:.4f}".format(epoch + 1, NUM_EPOCHS, 
                                                                                    avg_epoch_loss, 
                                                                                    avg_epoch_loss_discr, 
                                                                                    self.scheduler.get_last_lr()[0]))
            test = self.validate_on(self.model, self.test_loader, 
                                    self.reconstruct_loss, self.device, is_vae=True)

            self.train_loss[0].append(avg_epoch_loss)
            self.train_loss[1].append(avg_epoch_loss_discr)
            self.test_loss.append(test)

            # --- Check metrics and save models
            if (epoch % eval_freq == 0):
                self.eval_metrics(**kwargs)
                self.model_saving(avg_epoch_loss, epoch+1)

            self.scheduler.step()
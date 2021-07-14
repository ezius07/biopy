import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from .trainer import Trainer
from ..utils import StepClass
from ..models.ExprAutoEncoders import ClassDiscriminator


class SAAETrainer(Trainer):
    '''see readme'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader, **kwargs)
        self.train_loss = None
        self.test_loss = None
        self.reconstruct_loss = nn.MSELoss()  
        self.discriminate_loss = nn.CrossEntropyLoss()
        self.train_discriminator = True

    def generate_models_optimizers(self, model_class, discriminator_class=ClassDiscriminator, optimizer_SGD=False, **kwargs):
        super().generate_models_optimizers(model_class=model_class, optimizer_SGD=optimizer_SGD, **kwargs)

        # --- Collect parameters ---
        betas = self.parameters.get("BETAS", (0.9, 0.999))
        lr = self.parameters.get('LR', 1e-3)
        lr_d = self.parameters.get('LR_D', lr)
        wd = self.parameters.get('WEIGHT_DECAY', 1e-4)
        wd_d = self.parameters.get('WEIGHT_DECAY_D', wd)
        hidden_size = self.parameters['HIDDEN_SIZE']
        step_size = self.parameters['STEP_SIZE']
        gamma = self.parameters.get('GAMMA', 1)

        if kwargs.get('pretrained_discriminator', None) is None:

            n_labels = torch.unique(self.train_dataset.labels).shape[0]
            self.discriminator = discriminator_class(hidden_size, n_out=2, num_classes=n_labels).to(self.device)

            if optimizer_SGD == True:
                mom = self.parameters.get('MOMENTUM', 0.9)
                self.optimizer_discriminator = optim.SGD(self.discriminator.parameters(), lr=lr_d, 
                                                        momentum=mom, weight_decay=wd_d)
            else:
                self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr_d, weight_decay=wd_d)
            
            self.scheduler_discriminator = optim.lr_scheduler.StepLR(self.optimizer_discriminator, 
                                                                     step_size=step_size,
                                                                     gamma=gamma)

        else:
            print("[|||] Loading pretrained discriminator [|||]")
            self.discriminator = kwargs['pretrained_discriminator'].eval()
            self.optimizer_discriminator = None
            self.train_discriminator = False

    def train_model(self, sampler=None, **kwargs):
        if sampler is None:
            return Exception("You should specify a sampler")
        print("[|||] STARTED S-AAE TRAINING [|||]")
        cudnn.benchmark
        # --- Log losses ---
        self.train_loss = ([], [])
        self.test_loss = []

        # --- Collect parameters ---
        WEIGHT_DISCR = self.parameters.get('WEIGHT_DISCR', 2)
        MSE_WEIGHT = self.parameters.get('RECONSTRUCTION_WEIGHT', 1)
        ALPHA = self.parameters.get('ALPHA', 1)        
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 60)
        BATCH_SIZE = self.parameters.get('BATCH_SIZE', 32)
        metric_classes = kwargs.get('metric', 0)
        save_models = self.parameters.get('SAVE_MODELS', False)
        eval_freq = kwargs.get('eval_freq', 1)
        if not self.train_discriminator:
            WEIGHT_DISCR = self.parameters.get('PRETRAINED_DISCR_WEIGHT', WEIGHT_DISCR)

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

            for batch_features, labels in self.train_loader:
                batch_features = batch_features.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                if self.train_discriminator:
                    self.discriminator.train()
                    self.optimizer_discriminator.zero_grad()

                # compute reconstructions
                outputs, z, mean, log_var = self.model(batch_features)

                # real batch through discriminator
                discr_z_dist = self.discriminator(z, alpha=ALPHA, labels=labels)

                # sampled batch through discriminator 
                other_z, other_z_labels = sampler.sample(BATCH_SIZE, get_labels=True)
                other_z = other_z.to(self.device)
                other_z_labels = other_z_labels.to(self.device)
                discr_true_distr = self.discriminator(other_z, labels=other_z_labels)

                # compute loss
                mse_loss = self.reconstruct_loss(outputs, batch_features)
                z_dist_cse_loss = self.discriminate_loss(discr_z_dist, z_dist_label)
                true_dist_cse_loss = self.discriminate_loss(discr_true_distr, true_dist_label)

                loss = MSE_WEIGHT*mse_loss + WEIGHT_DISCR*(z_dist_cse_loss + true_dist_cse_loss)

                # compute accumulated gradients
                loss.backward()
                self.optimizer.step()
                if self.train_discriminator:
                    self.optimizer_discriminator.step()

                # add the mini-batch training loss to epoch loss
                avg_epoch_loss += mse_loss.item()
                avg_epoch_loss_discr += z_dist_cse_loss.item() + true_dist_cse_loss.item()

            # compute the epoch training loss
            avg_epoch_loss /=  len(self.train_loader)
            avg_epoch_loss_discr /= 2*len(self.train_loader)
            
            # display the epoch training loss
            if (epoch + 1) % self.parameters['LOG_FREQUENCY'] == 0:
                print("epoch : {}/{}, losses = ({:.4f}, {:.4f})  LR = {:.4f}".format(epoch + 1, NUM_EPOCHS,
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
            if self.train_discriminator:
                self.scheduler_discriminator.step()
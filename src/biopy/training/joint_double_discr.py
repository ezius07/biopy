import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn
from torch.utils.data import BatchSampler

from .joint_trainer import JointTrainer
from ..models.ExprAutoEncoders import Discriminator
from ..statistic import DistributionSampler
from ..utils import ThreadHandler
from ..utils import PredictableRandomSampler


class JointDoubleDiscr(JointTrainer):
    '''see readme file'''
    def __init__(self, parameters, omics=['mRNA', 'miRNA', 'meth27-450-preprocessed'], **kwargs):
        super().__init__(parameters, parameters['omics'])
        self.dataset = None
        self.model_args = None
        self.data_args = None
        self.datasets = {}
        self.input_sizes = {}
        self.train_datasets = {}
        self.test_datasets = {}
        self.train_loaders = {}
        self.test_loaders = {}
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        if 'omics' in parameters:
            self.omics = parameters['omics']
        else:          
            self.omics = omics
        self.n = len(self.omics)
        self.log_dir = kwargs.get('log_dir', None)
        self.train_loss = {}
        self.test_loss = {}
        self.metric_train = {}
        self.metric_test = {}
        
    def __generate_losses(self):
        self.reconstruct_loss = nn.MSELoss()
        self.discriminate_loss = nn.CrossEntropyLoss()
    
    def generate_models_optimizers(self, model_class, discriminator_class, optimizer_SGD=False, **kwargs):
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

        self.discr_labels = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=len(self.dataset.labels_ids)).to(self.device)
        self.discr_omics = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=self.n).to(self.device)

        if optimizer_SGD == True:
            mom = self.parameters.get('MOMENTUM', 0.9)
            self.optim_discr_labels = optim.SGD(self.discr_labels.parameters(), lr=lr_d, 
                                                    momentum=mom, weight_decay=wd_d)
            self.optim_discr_omics = optim.SGD(self.discr_omics.parameters(), lr=lr_d, 
                                                    momentum=mom, weight_decay=wd_d)

        else:
            self.optim_discr_labels = optim.Adam(self.discr_labels.parameters(), lr=lr_d, 
                                                      weight_decay=wd_d, betas=betas)
            self.optim_discr_omics = optim.Adam(self.discr_omics.parameters(), lr=lr_d, 
                                                      weight_decay=wd_d, betas=betas)
        
        self.schedul_discr_labels = optim.lr_scheduler.StepLR(self.optim_discr_labels, 
                                                                 step_size=step_size,
                                                                 gamma=gamma)
        self.schedul_discr_omics = optim.lr_scheduler.StepLR(self.optim_discr_omics, 
                                                                 step_size=step_size,
                                                                 gamma=gamma)

                
    def train_model(self, stage=1, **kwargs): 
        print("[|||] STARTING DOUBLE DISCRIMINATOR TRAINING [|||]")
        cudnn.benchmark
        # --- Log losses ---
        train_loss = ([], [])
        test_loss = []

        # --- Collect parameters ---
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 60)
        BATCH_SIZE = self.parameters.get('BATCH_SIZE', 32)
        WEIGHT_DISCR_OMICS = self.parameters.get('WEIGHT_DISCR_OMICS', 2)
        WEIGHT_DISCR_LABELS = self.parameters.get('WEIGHT_DISCR_LABELS', 1)
        MSE_WEIGHT = self.parameters.get('RECONSTRUCTION_WEIGHT', 1)
        ALPHA = self.parameters.get('ALPHA', 1)
        log_freq = self.parameters.get('LOG_FREQUENCY', 3)
        metric_classes = kwargs.get('metric', 0)
        save_models = self.parameters.get('SAVE_MODELS', False)
        eval_freq = kwargs.get('eval_freq', 1)

        # --- Set up discriminator labels ---
        dict_label = {omic: i*torch.ones(BATCH_SIZE).long().to(self.device) for i, omic in enumerate(self.omics)}

        # --- Set up metrics ---
        self.setup_metrics(metric_classes)

        # --- Set up model saving ---
        self.setup_model_saving(save_models)

        #n_batches = len(self.train_loaders[self.omics[0]])
        #threader = ThreadHandler(list(self.train_loaders.values()))
        #threader.setup_threads()
        #threader.start_threads()
        ordered_loaders = [self.train_loaders[omic] for omic in self.omics]
        n_batches = min(map(len, ordered_loaders))

        for epoch in range(NUM_EPOCHS):
            avg_epoch_loss = 0
            avg_epoch_loss_discr = 0
            loaders = list(map(iter, ordered_loaders)) # Starts the dataloader in background
            for _ in range(n_batches):
                #batches = [threader.get_from(i)[0].to(self.device) for i in range(self.n)]
                #batches = [next(loader)[0].to(self.device) for loader in loaders]
                batches = [next(loader) for loader in loaders]
                batches = [(feats.to(self.device), labs.to(self.device)) for feats, labs in batches]

                for i, omic in enumerate(self.omics): 
                    self.models[omic].train()
                    self.optimizers[omic].zero_grad()

                    self.discr_labels.train()
                    self.optim_discr_labels.zero_grad()

                    self.discr_omics.train()
                    self.optim_discr_omics.zero_grad()
    
                    # compute reconstructions
                    outputs, z, mean, log_var = self.models[omic](batches[i][0])

                    # batch through labels discriminator
                    label_scores = self.discr_labels(z)

                    # batch through omics discriminator
                    omic_scores = self.discr_omics(z, alpha=ALPHA)

                    # compute loss
                    mse_loss = self.reconstruct_loss(outputs, batches[i][0])
                    labels_loss = self.discriminate_loss(label_scores, batches[i][1])
                    omics_loss = self.discriminate_loss(omic_scores, dict_label[omic])

                    loss = MSE_WEIGHT*mse_loss + WEIGHT_DISCR_LABELS*labels_loss + WEIGHT_DISCR_OMICS*omics_loss

                    # compute accumulated gradients
                    loss.backward()
                    self.optimizers[omic].step()
                    self.optim_discr_labels.step()
                    self.optim_discr_omics.step()

                    # add the mini-batch training loss to epoch loss
                    avg_epoch_loss += mse_loss.item()
                    avg_epoch_loss_discr += labels_loss.item() + omics_loss.item()


            # compute the epoch training loss
            avg_epoch_loss /= n_batches
            avg_epoch_loss_discr /= 2 * n_batches
            # display the epoch training loss
            if (epoch + 1) % log_freq == 0:
                print("epoch : {}/{}, sum losses = ({:.4f}, {:.4f}), LR = {:.4f}".format(epoch + 1, NUM_EPOCHS,
                                                                         avg_epoch_loss, avg_epoch_loss_discr,
                                                                         self.schedulers[omic].get_last_lr()[0]))
                
            test = 0
            for omic in self.omics:
                test += self.validate_on(self.models[omic], self.test_loaders[omic], 
                                         self.reconstruct_loss, self.device, is_vae=True)
            
            train_loss[0].append(avg_epoch_loss)
            train_loss[1].append(avg_epoch_loss_discr)
            test_loss.append(test)
            
            for omic in self.omics:
                self.schedulers[omic].step()
            self.schedul_discr_labels.step()
            self.schedul_discr_omics.step()

            # --- Check metrics and save models
            if (epoch % eval_freq == 0):
                self.eval_metrics(**kwargs)
                self.model_saving(avg_epoch_loss, epoch+1)
        
        self.train_loss[0] = train_loss
        self.test_loss[0] = test_loss
        self.metric_train = {0:self.metric_train}
        self.metric_test = {0:self.metric_test}
        self.metric_names = {0:self.metric_names}
        #threader.terminate_threads()

    @staticmethod
    def get_trainer_for_model(ae_class):
        return JointDoubleDiscr
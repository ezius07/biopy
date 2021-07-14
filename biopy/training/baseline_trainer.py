import torch
from torch import nn
from torch import optim
from torch.backends import cudnn

from .trainer import Trainer
from .joint_trainer import JointTrainer
from itertools import combinations


class VAEPzEstimationTrainer(Trainer):
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None,
                 **kwargs):
        super().__init__(parameters, model, optimizer, scheduler, train_loader, test_loader)
        self.train_loss = None
        self.test_loss = None
        self.log_dir = kwargs.get('log_dir', None)
        self.reconstruct_loss = nn.MSELoss()
        self.discriminate_loss = nn.CrossEntropyLoss()

    def generate_models_optimizers(self, model_class, discriminator_class, optimizer_SGD=False, **kwargs):
        super().generate_models_optimizers(model_class=model_class, optimizer_SGD=optimizer_SGD, **kwargs)

        self.discriminator = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=len(self.dataset.labels_ids)).to(
            self.device)

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.parameters.get('LR_D', self.parameters['LR']),
                                                  weight_decay=self.parameters.get('WEIGHT_DECAY_D',
                                                                                   self.parameters['WEIGHT_DECAY']),
                                                  betas=self.parameters.get("BETAS", (0.9, 0.999)))

    def train_model(self, reconstruct_loss=None, **kwargs):
        """
           This reproduces the first step of training where P_z (the distribution of the latent space)
           is estimated using a single-omic training and the losses are reconstruction+kld+
           a discriminator that uses cross entropy loss over class labels on the features of the latent 
        """

        if reconstruct_loss is not None:
            self.reconstruct_loss = reconstruct_loss

        cudnn.benchmark
        # --- Log losses ---
        self.train_loss = ([], [])
        self.test_loss = []
        
        # --- Collect parameters ---
        weight_kld = self.parameters.get('KLD_WEIGHT', 1e-8)
        weight_discriminative = self.parameters.get('DISCRIMINATIVE_WEIGHT', 1)
        weight_reconstruction = self.parameters.get('RECONSTRUCTION_WEIGHT', 0.1)
        save_models = self.parameters.get('SAVE_MODELS', False)
        metric_classes = kwargs.get('metric', 0)
        eval_freq = kwargs.get('eval_freq', 1)

        # --- Set up metrics ---
        self.setup_metrics(metric_classes)

        # --- Set up model saving ---
        self.setup_model_saving(save_models)

        print("[|||] STARTING TRAINING of VAEPzEstimationTrainer [|||]")

        for epoch in range(self.parameters['NUM_EPOCHS']):
            avg_epoch_loss = 0
            avg_epoch_loss_discr = 0

            correct = 0

            for batch_features, labels in self.train_loader:
                batch_features = batch_features.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                self.discriminator.train()
                self.optimizer_discriminator.zero_grad()

                # compute reconstrunctions and output of internal classifier on latent features
                outputs, latents, mean, log_var = self.model(batch_features)

                # compute KLD loss and reconstruction loss (default MSE)
                mse_loss = self.reconstruct_loss(outputs, batch_features)
                kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = weight_reconstruction * mse_loss + weight_kld * kld_loss

                # compute discrimination loss on predictions on latent features
                outputs_latents = self.discriminator(latents)

                ###############################################
                _, predicted = torch.max(outputs_latents.data, 1)
                n_corrects = int((predicted == labels).sum())
                correct += n_corrects
                ################################################

                loss_discr = self.discriminate_loss(outputs_latents, labels)
                loss += weight_discriminative * loss_discr

                avg_epoch_loss += mse_loss.item()
                avg_epoch_loss_discr += loss_discr.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer_discriminator.step()

            avg_epoch_loss /= len(self.train_loader)
            avg_epoch_loss_discr /= len(self.train_loader)
            acc = correct / (len(self.train_loader) * self.parameters["BATCH_SIZE"])

            if (epoch + 1) % self.parameters['LOG_FREQUENCY'] == 0:
                print(
                    "epoch : {}/{}, train_acc = {:.4f}, train_loss = {:.4f}, train_discr_loss = {:.4f}, LR = {:.4f}".format(
                        epoch + 1, self.parameters['NUM_EPOCHS'], acc, avg_epoch_loss,
                        avg_epoch_loss_discr, self.scheduler.get_last_lr()[0]))
            test = self.validate_on(self.model, self.test_loader, self.reconstruct_loss, self.device, is_vae=True)

            # --- Check metrics and save models
            if epoch % eval_freq == 0:
                self.eval_metrics(**kwargs)
                self.model_saving(avg_epoch_loss, epoch + 1)
            
            self.train_loss[0].append(avg_epoch_loss)
            self.train_loss[1].append(avg_epoch_loss_discr)
            self.test_loss.append(test)
            self.scheduler.step()

        return self.model, self.dataset.omic_selected

    @staticmethod
    def get_trainer_for_model(ae_class):
        return VAEPzEstimationTrainer


class VAETrainer(JointTrainer):
    '''see readme file'''
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, parameters['omics'], **kwargs)

        self.train_loss = None
        self.test_loss = None
        self.metric_train = None
        self.metric_test = None
        self.log_dir = kwargs.get('log_dir', None)
        self.anchor_loss = nn.L1Loss()

    def generate_models_optimizers(self, model_class, discriminator_class, optimizer_SGD=False, **kwargs):
        super().generate_models_optimizers(model_class=model_class, optimizer_SGD=optimizer_SGD, **kwargs)

        self.discriminator = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=len(self.dataset.labels_ids)).to(self.device)

        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.parameters.get('LR_D', self.parameters['LR']),
                                                  weight_decay=self.parameters.get('WEIGHT_DECAY_D',
                                                                                   self.parameters['WEIGHT_DECAY']),
                                                  betas=self.parameters.get("BETAS", (0.9, 0.999)))

    def train_model(self, result=None, reconstruct_loss=None, **kwargs):
        # Add previously trained model for the selected omic
        if result is not None:
            prev_model, prev_omic_selected = result
            # by substituting this model the model of the previous omic is not trained if present in this stage
            self.models[prev_omic_selected] = prev_model

        if reconstruct_loss is not None:
            self.reconstruct_loss = reconstruct_loss
        
        cudnn.benchmark
        # --- Log losses ---
        self.train_loss = ([], [])
        self.test_loss = []
            
        # --- Collect parameters ---
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 60)
        weight_discriminative = self.parameters.get('DISCRIMINATIVE_WEIGHT', 10)
        weight_reconstruction = self.parameters.get('RECONSTRUCTION_WEIGHT', 10)
        weight_kld = self.parameters.get('KLD_WEIGHT', 0)
        metric_classes = kwargs.get('metric', 0)
        save_models = self.parameters.get('SAVE_MODELS', False)
        weight_anchor = self.parameters.get('ANCHOR_WEIGHT', 0.1)
        eval_freq = kwargs.get('eval_freq', 1)

        
        # --- Set up metrics ---
        self.setup_metrics(metric_classes)
        
        # --- Set up model saving ---
        self.setup_model_saving(save_models)


        print("[|||] STARTING TRAINING of Joint VAETrainer [|||]")        

        ordered_loaders = [self.train_loaders[omic] for omic in self.omics]
        n_batches = min(map(len, ordered_loaders))

        for epoch in range(NUM_EPOCHS):
            avg_epoch_loss = 0
            avg_epoch_loss_discr = 0

            loaders = list(map(iter, ordered_loaders))  # Starts the dataloader in background
            for _ in range(n_batches):
                # batches = [threader.get_from(i)[0].to(self.device) for i in range(self.n)]
                batches = [next(loader) for loader in loaders]
                # print([b[:][1] for b in batches]) # Debug print of labels
                self.optimizer_discriminator.zero_grad()

                latents = {}
                self.discriminator.eval()
                # Train encoder-decoder
                for i, omic in enumerate(self.omics):
                    self.models[omic].train()
                    self.optimizers[omic].zero_grad()

                    # compute reconstrunctions and latent features
                    outputs, latents[omic], mean, log_var = self.models[omic](batches[i][0].to(self.device))

                    # compute reconstruction loss (default MSE)
                    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    mse_loss = self.reconstruct_loss(outputs, batches[i][0].to(self.device))

                    loss = weight_reconstruction * mse_loss + weight_kld * kld_loss

                    # compute discrimination loss on predictions on latent features
                    outputs_latents = self.discriminator(latents[omic])
                    loss_discr = self.discriminate_loss(outputs_latents, batches[i][1].to(self.device))
                    loss += weight_discriminative * loss_discr

                    avg_epoch_loss += mse_loss.item()
                    avg_epoch_loss_discr += loss_discr.item()

                    loss.backward(retain_graph=True)

                loss_anchor = 0
                for omic1, omic2 in combinations(self.omics, 2):
                    loss_anchor += self.anchor_loss(latents[omic1], latents[omic2])
                loss_anchor *= weight_anchor
                loss_anchor.backward()

                for omic in self.omics:
                    # Warning: the prev_model does not get optimized!
                    self.optimizers[omic].step()

                # Train NON-adversarial clf
                for i, omic in enumerate(self.omics):
                    self.models[omic].eval()

                    self.discriminator.train()
                    self.optimizer_discriminator.zero_grad()

                    _, latents, _, _ = self.models[omic](batches[i][0].to(self.device))

                    # compute discrimination loss on predictions on latent features
                    outputs_latents = self.discriminator(latents)
                    loss_discr = self.discriminate_loss(outputs_latents, batches[i][1].to(self.device))
                    loss = weight_discriminative * loss_discr

                    avg_epoch_loss_discr += loss_discr.item()

                    loss.backward()

                    self.optimizer_discriminator.step()

            avg_epoch_loss /= n_batches
            avg_epoch_loss_discr /= 2 * n_batches

            if (epoch + 1) % self.parameters['LOG_FREQUENCY'] == 0:
                print("epoch : {}/{}, train_loss = {:.4f}, train_discr_loss = {:.4f}".format(epoch + 1, self.parameters[
                    'NUM_EPOCHS'], avg_epoch_loss, avg_epoch_loss_discr))

            test = 0
            for omic in self.omics:
                test += self.validate_on(self.models[omic], self.test_loaders[omic], self.reconstruct_loss, self.device,
                                         is_vae=True)


            self.train_loss[0].append(avg_epoch_loss)
            self.train_loss[1].append(avg_epoch_loss_discr)
            self.test_loss.append(test)

            for omic in self.omics:
                self.schedulers[omic].step()
            
            # --- Check metrics and save models
            if (epoch % eval_freq == 0):
                self.eval_metrics(**kwargs)
                self.model_saving(avg_epoch_loss, epoch+1)
            
        return (self.model, self.dataset.omic_selected)

    def pack_for_metric(self, metric_class, split, **kwargs):
        if split == 'train':
            dataset = self.train_datasets
        elif split == 'test':
            dataset = self.test_datasets

        kwargs["train_dataset"] = self.train_datasets
        aaes = self.models
        metric = metric_class(dataset, aaes, self.device, **kwargs)

        return metric

    @staticmethod
    def get_trainer_for_model(ae_class):
        return VAETrainer
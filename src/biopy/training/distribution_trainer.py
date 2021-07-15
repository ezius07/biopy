import torch
from torch import nn
from torch import optim
from torch.backends import cudnn

from .joint_trainer import JointTrainer


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class VAETrainerDistribution(JointTrainer):
    """see readme file"""

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

        self.discriminator = discriminator_class(self.parameters['HIDDEN_SIZE'], n_out=len(self.dataset.labels_ids)).to(
            self.device)

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
        eval_freq = kwargs.get('eval_freq', 1)

        # parameters HAFN
        weight_hafn = self.parameters.get('HAFN_WEIGHT', 0)
        hafn_radius = self.parameters.get('HAFN_RADIUS', 1)
        # parameters SAFN
        weight_safn = self.parameters.get('SAFN_WEIGHT', 0)
        # parameters MMD
        weight_mmd = self.parameters.get('MMD_WEIGHT', 0)

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

                    if weight_hafn != 0:
                        # HAFN loss
                        hafn_loss = (latents[omic].norm(p=2, dim=1).mean() - hafn_radius) ** 2
                        loss += weight_hafn * hafn_loss

                    if weight_safn != 0:
                        # SAFN loss
                        radius = latents[omic].norm(p=2, dim=1).detach()
                        assert radius.requires_grad == False
                        radius = radius + 1.0
                        safn_loss = ((latents[omic].norm(p=2, dim=1) - radius) ** 2).mean()
                        loss += weight_safn * safn_loss

                    if weight_mmd != 0:
                        # MMD loss
                        if self.omics[(i-1) % self.n] in latents:
                            loss += weight_mmd * mmd(latents[omic], latents[self.omics[(i-1) % self.n]])

                    # compute discrimination loss on predictions on latent features
                    outputs_latents = self.discriminator(latents[omic])
                    loss_discr = self.discriminate_loss(outputs_latents, batches[i][1].to(self.device))
                    loss += weight_discriminative * loss_discr

                    avg_epoch_loss += mse_loss.item()
                    avg_epoch_loss_discr += loss_discr.item()

                    loss.backward(retain_graph=True)

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
                self.model_saving(avg_epoch_loss, epoch + 1)

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
        return VAETrainerDistribution

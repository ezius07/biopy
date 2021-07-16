import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn
from torch.utils.data import BatchSampler

from .trainer import Trainer
from ..models.ExprAutoEncoders import Discriminator
from ..statistic import DistributionSampler
from ..utils import ThreadHandler
from ..utils import PredictableRandomSampler


class JointTrainer(Trainer):
    '''see readme file'''
    def __init__(self, parameters, omics=['mRNA', 'miRNA', 'meth27-450-preprocessed'], **kwargs):
        super().__init__(parameters)
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
        print("----Parameters: -----")
        print(parameters.data)
        print("--------------")
        
    def __generate_losses(self):
        self.reconstruct_loss = nn.MSELoss()
        self.discriminate_loss = nn.CrossEntropyLoss()
            
    def generate_dataset_loaders(self, dataset_class, **kwargs):
        self.data_args = (dataset_class, kwargs)
        print("[|||] INSTANTIATING DATASET [|||]")
        self.dataset = dataset_class(**kwargs)

        print("[|||] PREPROCESSING DATASET [|||]")
        # Preprocessing acts in-place in the dataset object
        self.preprocess_dataset(self.dataset, self.parameters, all_omics=True)
        
        for omic in self.omics:
            print(f"[|||] LOADING {omic} DATA [|||]")
            self.datasets[omic] = self.dataset.set_omic(omic)
            self.input_sizes[omic] = self.datasets[omic][0][0].shape[0]

            self.train_datasets[omic], self.test_datasets[omic] = self.datasets[omic].train_val_test_split()

            self.train_loaders[omic] = data.DataLoader(self.train_datasets[omic],
                                                      batch_sampler=BatchSampler(PredictableRandomSampler(len(self.train_datasets[omic])), batch_size=self.parameters['BATCH_SIZE'], drop_last=True), 
                                                      num_workers=4, pin_memory=True)

            self.test_loaders[omic] = data.DataLoader(self.test_datasets[omic], batch_size=64, shuffle=False, 
                                                     num_workers=4)
            
            
    def generate_models_optimizers(self, model_class, discriminator_class=Discriminator, optimizer_SGD=False, **kwargs):
        if len(self.input_sizes) == 0:
            raise ValueError("Either you call 'generated_datasets' first, or you have to provide input_sizes")  
        
        self.model_args = (model_class, discriminator_class, optimizer_SGD, kwargs)

        # --- Collect parameters ---
        betas = self.parameters.get("BETAS", (0.9, 0.999))
        lr = self.parameters.get('LR', 1e-3)
        lr_d = self.parameters.get('LR_D', lr)
        wd = self.parameters.get('WEIGHT_DECAY', 1e-4)
        wd_d = self.parameters.get('WEIGHT_DECAY_D', wd)
        hidden_size = self.parameters['HIDDEN_SIZE']
        step_size = self.parameters['STEP_SIZE']
        gamma = self.parameters.get('GAMMA', 1)

        print("[|||] BUILDING MODELS [|||]")
        for omic in self.omics:
            input_size = self.input_sizes[omic]
            n_hidden = kwargs.get("n_hidden", input_size)

            self.models[omic] = model_class(data_size=input_size, hidden_size=hidden_size, 
                                            n_hidden=n_hidden, **kwargs).to(self.device)

            if optimizer_SGD == True:
                mom = self.parameters.get('MOMENTUM', 0.9)
                self.optimizers[omic] = optim.SGD(self.models[omic].parameters(), lr=lr, 
                                                  momentum=mom, weight_decay=wd)
            else:
                self.optimizers[omic] = optim.Adam(self.models[omic].parameters(), lr=lr, 
                                                  weight_decay=wd, betas=betas)

            self.schedulers[omic] = optim.lr_scheduler.StepLR(self.optimizers[omic], 
                                                              step_size=step_size,
                                                              gamma=gamma)

        #n_labels = torch.unique(self.train_datasets[self.omics[0]].labels).shape[0]
        self.discriminator = discriminator_class(hidden_size, n_out=self.n, num_classes=len(self.dataset.labels_ids)).to(self.device)
        
        if optimizer_SGD == True:
            mom = self.parameters.get('MOMENTUM', 0.9)
            self.optimizer_discriminator = optim.SGD(self.discriminator.parameters(), lr=lr_d, 
                                                    momentum=mom, weight_decay=wd_d)
        else:
            self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr_d, 
                                                      weight_decay=wd_d, betas=betas)
        
        self.scheduler_discriminator = optim.lr_scheduler.StepLR(self.optimizer_discriminator, 
                                                                 step_size=step_size,
                                                                 gamma=gamma)
        self.__generate_losses()
    

    def train_model(self, stage=1, model_omic="mRNA", **kwargs):
        #if self.model_args[0].ae_type not in ['VAE', 'AAE', 'SAAE']:
        #    raise Exception("This trainer requires a model that is able to impose a distribution")

        if stage == 1:
            print("[|||] STARTING STAGE 1 [|||]")
            self.__joint_training(**kwargs)
            self.models = {stage: self.models}

        elif stage == 2:
            print("[|||] STARTING STAGE 2 [|||]")
            self.__single_trainings(model_omic, **kwargs)

    def __single_trainings(self, model_omic, **kwargs):
        stage = 2
        self.models[stage] = {}
        self.metric_names[1] = []

        sampler=DistributionSampler()
        sampler.build_empirical_from_latent(self.models[1][model_omic], self.datasets[model_omic])
        trainer_class = self.get_trainer_for_second_step(self.model_args[0]) 
        data_args = self.data_args[1]
        dataset_class = self.data_args[0]
        
        for omic in self.omics:
            print(f"[|||] {omic} [|||]")
            data_args.pop('omics', -1)
            data_args['omics'] = [omic]

            trainer = trainer_class(self.parameters, log_dir=os.path.join(self.log_dir, omic))
            trainer.parameters.update({'INPUT_SIZE': self.input_sizes[omic]})
            # --- Avoid reinstantiating datasets ---
            trainer.train_dataset = self.train_datasets[omic]
            trainer.test_dataset = self.test_datasets[omic]            
            trainer.train_loader = self.train_loaders[omic]
            trainer.test_loader = self.test_loaders[omic]

            trainer.generate_models_optimizers(self.model_args[0], self.model_args[1], self.model_args[2], 
                                               pretrained_discriminator=self.discriminator, **self.model_args[3])
            trainer.train_model(sampler, **kwargs)

            # --- update models and losses ---
            self.models[stage][omic] = trainer.model
            self.metric_names[1] += [f"{omic}_{metric_name}" for metric_name in trainer.metric_names]
            self.__update_loss_from_subtrainer(trainer.train_loss, trainer.test_loss)
            self.__update_metrics_from_subtrainer(trainer.metric_train, trainer.metric_test, omic)
        

        self.train_loss[1] = (list(self.train_loss[1][0]), 
                              list(self.train_loss[1][1]))
        self.test_loss[1] = list(self.test_loss[1])
        
        if trainer.metric_names[0] != 'mse_loss':
            metric_list = []
            for key in self.metric_train[1]:
                metric_list += list(self.metric_train[1][key])
            self.metric_train[1] = tuple(metric_list)

            metric_list = []
            for key in self.metric_test[1]:
                metric_list += list(self.metric_test[1][key])
            self.metric_test[1] = tuple(metric_list)

    def __joint_training(self, **kwargs):
        cudnn.benchmark
        # --- Log losses ---
        train_loss = ([], [])
        test_loss = []

        # --- Collect parameters ---
        NUM_EPOCHS = self.parameters.get('NUM_EPOCHS', 60)
        BATCH_SIZE = self.parameters.get('BATCH_SIZE', 32)
        WEIGHT_DISCR = self.parameters.get('WEIGHT_DISCR', 2)
        MSE_WEIGHT = self.parameters.get('RECONSTRUCTION_WEIGHT', 1)
        ALPHA = self.parameters.get('ALPHA', 1)
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

        print("[|||] STARTING JOINT TRAINING [|||]")
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

                    self.discriminator.train()
                    self.optimizer_discriminator.zero_grad()
    
                    # compute reconstructions
                    outputs, z, mean, log_var = self.models[omic](batches[i][0])
                    # real batch through discriminator
                    discr_z_dist = self.discriminator(z, alpha=ALPHA, labels=batches[i][1])

                    # sampled batch through discriminator
                    with torch.no_grad():
                        other_z = self.models[self.omics[(i+1) % self.n]].encode_and_sample(batches[(i+1) % self.n][0])
                    discr_true_distr = self.discriminator(other_z, labels=batches[(i+1) % self.n][1])

                    # compute loss
                    mse_loss = self.reconstruct_loss(outputs, batches[i][0])
                    z_dist_cse_loss = self.discriminate_loss(discr_z_dist, dict_label[omic])
                    true_dist_cse_loss = self.discriminate_loss(discr_true_distr, dict_label[self.omics[(i+1) % self.n]])

                    loss = MSE_WEIGHT*mse_loss + WEIGHT_DISCR*(z_dist_cse_loss + true_dist_cse_loss)
                    # compute accumulated gradients
                    loss.backward()
                    self.optimizers[omic].step()
                    self.optimizer_discriminator.step()

                    # add the mini-batch training loss to epoch loss
                    avg_epoch_loss += mse_loss.item()
                    avg_epoch_loss_discr += z_dist_cse_loss.item() + true_dist_cse_loss.item()


            # compute the epoch training loss
            avg_epoch_loss /= n_batches
            avg_epoch_loss_discr /= 2 * n_batches
            # display the epoch training loss
            if (epoch + 1) % self.parameters.get('LOG_FREQUENCY', 3) == 0:
                print("epoch : {}/{}, sum loss over all omics = ({:.4f}, {:.4f}), LR = {:.4f}".format(epoch + 1, NUM_EPOCHS,
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
            self.scheduler_discriminator.step()

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
    
    def pack_for_metric(self, metric_class, split, **kwargs):
        if split == 'train':
            dataset = self.train_datasets
        elif split == 'test':
            dataset = self.test_datasets

        kwargs["train_dataset"] = self.train_datasets
        aaes = self.models
        metric = metric_class(dataset, aaes, self.device, **kwargs)

        return metric

    def __update_metrics_from_subtrainer(self, train_metric, test_metric, omic):
        stage = 1
        if train_metric is not None:
            if stage not in self.metric_train:
                self.metric_train[stage] = {omic: train_metric} 
                self.metric_test[stage] = {omic: test_metric}
            else:
                self.metric_train[stage][omic] = train_metric
                self.metric_test[stage][omic] = test_metric

    def __update_loss_from_subtrainer(self, train_loss, test_loss):
        stage = 1
        if stage not in self.train_loss:
            self.train_loss[stage] = [np.array(train_loss[0]), 
                                      np.array(train_loss[1])]
            self.test_loss[stage] = np.array(test_loss)
        else:
            for i, loss in enumerate(train_loss):
                self.train_loss[stage][i] += np.array(loss)
            self.test_loss[stage] += np.array(test_loss)

    def setup_metrics(self, metric_classes):
        self.metric_classes = metric_classes if isinstance(metric_classes, list) else [metric_classes]
        self.metric_names = [m.__name__ if m != 0 else 'mse_loss' for m in self.metric_classes]
        
        if self.metric_classes[0] != 0:
            for metric in metric_classes:
                if metric.separate_omics:
                    initial_index = self.metric_names.index(metric.__name__)
                    del self.metric_names[initial_index]
                    for i, omic in enumerate(self.omics):
                        self.metric_names.insert(i+initial_index, metric.__name__ + "_" + omic)
                    self.metric_names.insert(len(self.omics)+initial_index, metric.__name__)
            n_metrics = len(self.metric_names)
            self.metric_train = tuple([] for _ in range(n_metrics))
            self.metric_test = tuple([] for _ in range(n_metrics))

    def eval_metrics(self, **kwargs):
        index = 0
        if (self.metric_classes[0] != 0):
            metric_on_test = {}
            for metric in self.metric_classes:
                if metric.supports_train:
                    metric_train = self.pack_for_metric(metric_class=metric, split='train', **kwargs)
                    # if the metric can be calculated separately on single omics this is done
                    # but also the calculation with all omics together is done
                    if metric.separate_omics:
                        for i, omic in enumerate(self.omics):
                            metric_on_train = metric_train(omics_train=omic, omics_test=omic, **kwargs)
                            self.metric_train[index+i].append(metric_on_train)

                    metric_on_train = metric_train(**kwargs)
                    self.metric_train[index].append(metric_on_train)

                metric_test = self.pack_for_metric(metric_class=metric, split='test', **kwargs)
                if metric.separate_omics:
                    for omic in self.omics:
                        metric_on_test[self.metric_names[index]] = metric_test(omics_train=omic, omics_test=omic, **kwargs)
                        self.metric_test[index].append(metric_on_test[self.metric_names[index]])
                        index += 1

                metric_on_test[self.metric_names[index]] = metric_test(**kwargs)
                self.metric_test[index].append(metric_on_test[self.metric_names[index]])
                index += 1
            self.metric_on_test = metric_on_test

    def setup_model_saving(self, save_models):
        self.save_models = save_models
        if save_models:
            if self.log_dir is None:
                raise Exception("You asked for model saving but didn't specify log_dir")
            for name in self.metric_names:
                os.makedirs(os.path.join(self.log_dir, name), exist_ok=True)

            self.best_scores = [0 if metric != 'mse_loss' else 1e9 for metric in self.metric_names]
            self.comparators = [max if metric != 'mse_loss' else min for metric in self.metric_names]
            self.last_best_file = [{omic: "" for omic in self.omics} for metric in self.metric_names]

    def model_saving(self, avg_epoch_loss, epoch):
        if self.save_models:
            scores = [self.metric_on_test[metric] if metric != "mse_loss" else avg_epoch_loss for metric in self.metric_names]
            for i, metric in enumerate(self.metric_names):
                if self.comparators[i](self.best_scores[i], scores[i]) == scores[i]:
                    self.best_scores[i] = scores[i]
                    for omic in self.omics:
                        if self.last_best_file[i][omic] != "":
                            os.remove(self.last_best_file[i][omic])

                        torch.save(self.models[omic].state_dict(),
                                   os.path.join(self.log_dir, metric,
                                                omic + self.__class__.__name__ + f"epoch{epoch}.pth"))
                        self.last_best_file[i][omic] = os.path.join(self.log_dir, metric, 
                                                                   omic + self.__class__.__name__ + f"epoch{epoch}.pth")
               
    @staticmethod
    def get_trainer_for_second_step(ae_class):
        ae_type = ae_class.ae_type
        trainer = None
        if ae_type == 'VAE' or ae_type == 'AAE':
            from .aae_trainer import AAETrainer
            trainer = AAETrainer
        elif ae_type == 'SAAE':
            from .saae_trainer import SAAETrainer
            trainer = SAAETrainer 

        return trainer

    @staticmethod
    def get_trainer_for_model(ae_class):
        return JointTrainer
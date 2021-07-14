import os
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.backends import cudnn

from ..utils import StepClass

class Trainer():
    '''see readme file'''
    def __init__(self, parameters, model=None, optimizer=None, scheduler=None, train_loader=None, test_loader=None, log_dir=None, **kwargs):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = parameters
        self.dataset = None
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader 
        self.log_dir = log_dir
        # ---- Metrics
        self.metric_classes = None
        self.metric_names = None
        self.metric_train = None
        self.metric_test = None
        # ---- Model check
        self.save_models = False
        best_scores = None
        comparators = None
        last_best_file = None
            
    # It is better to execute the generate_dataset_loader first in order to infer the INPUT_SIZE
    def generate_models_optimizers(self, model_class, optimizer_SGD=False, **kwargs):
        # create a model and load it to the specified device
        hidden_size = self.parameters['HIDDEN_SIZE']
        input_size = self.parameters.get('INPUT_SIZE', 0)
        betas = self.parameters.get("BETAS", (0.9, 0.999))
        if input_size == 0:
            raise Exception("Either you call 'generated_datasets' first, or you have to provide INPUT_SIZE")
        print("[|||] BUILDING MODEL [|||]")
        self.model = model_class(data_size=input_size, hidden_size=hidden_size, **kwargs).to(self.device)

        # create an optimizer object
        if optimizer_SGD is False:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameters['LR'], 
                                        weight_decay=self.parameters['WEIGHT_DECAY'], betas=betas)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.parameters['LR'], 
                                       momentum=self.parameters['MOMENTUM'], 
                                       weight_decay=self.parameters['WEIGHT_DECAY'])

        # scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.parameters['STEP_SIZE'], 
                                                  gamma=self.parameters['GAMMA'])
    
    def generate_dataset_loaders(self, dataset_class, **args):
        print("[|||] INSTANTIATING DATASET [|||]")
        print(f"[|||] LOADING DATA [|||]")
        self.dataset = dataset_class(**args)

        print("[|||] PREPROCESSING DATASET [|||]")
        # Preprocessing acts in-place in the dataset object
        # Trainer class works on singe-omic dataset
        self.preprocess_dataset(self.dataset, self.parameters, all_omics=False)

        self.train_dataset, self.test_dataset = self.dataset.train_val_test_split()

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=self.parameters['BATCH_SIZE'], 
                                            shuffle=True, num_workers=4, 
                                            pin_memory=True, drop_last=True)
        
        self.test_loader = data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        if 'INPUT_SIZE' not in self.parameters:
            self.parameters['INPUT_SIZE'] = self.train_dataset[0][0].shape[0]


    def pack_for_metric(self, metric_class, split, **kwargs):
        if split == 'train':            
            dataset = {self.train_dataset.omic_selected: self.train_dataset}
        elif split == 'test':
            dataset = {self.test_dataset.omic_selected: self.test_dataset}

        aaes = {self.test_dataset.omic_selected: self.model}
        kwargs["train_dataset"] = {self.train_dataset.omic_selected: self.train_dataset}
        metric = metric_class(dataset, aaes, self.device, **kwargs)

        return metric
    
    @staticmethod
    def preprocess_dataset(dataset, parameters, all_omics=False):
        def minmax():
            dataset.minmax_scale(all_omics=all_omics)

        def standardize():
            dataset.standardize(all_omics=all_omics)

        def minmax_fs_std():
            dataset.minmax_scale(all_omics=all_omics)
            dataset.feature_selection_by('std', top_n=parameters.get('PREPROCESS_FS_TOP_N', 300), all_omics=all_omics)

        def standardize_fs_mad():
            dataset.standardize(all_omics=all_omics)
            dataset.feature_selection_by('mad', top_n=parameters.get('PREPROCESS_FS_TOP_N', 300), all_omics=all_omics)

        def identity():
            pass
        
        # List of omics to preprocess with log1p
        if 'PREPROCESS_LOG1P' in parameters:
            original_omic_selected = dataset.omic_selected
            for omic_name in parameters['PREPROCESS_LOG1P']:
                dataset.set_omic(omic_name)
                dataset.log1p(all_omics=False)
            dataset.set_omic(original_omic_selected)

        preprocessor = locals()[parameters.get('PREPROCESS', 'standardize')]
        preprocessor() # Preprocessing functions act in-place in the dataset object

        if parameters.get('PREPROCESS_SMOTE', False):
            dataset.SMOTE(all_omics=all_omics)

    @staticmethod
    @torch.no_grad()
    def validate_on(net, data_loader, criterion, device, is_vae=False):
        '''given a dataloader and a network performs an evaluation pass'''
        net.train(False) # Set Network to evaluation mode
        avg_loss = 0.0
        running_corrects = 0
        for input_features, _ in data_loader:
            input_features = input_features.to(device)
            
            # Forward Pass
            if is_vae:
                outputs = net(input_features)
                output = outputs[0]
            else:
                outputs = net(input_features)
            
            loss = criterion(output, input_features).item()

            avg_loss += loss                
        # Calculate Accuracy
        avg_loss = avg_loss / len(data_loader)
        
        return avg_loss

    @staticmethod
    def get_trainer_for_model(ae_class):
        ae_type = ae_class.ae_type
        trainer = None
        if ae_type == 'VAE':
            from .vae_trainer import VAETrainer
            trainer = VAETrainer
        elif ae_type == 'AAE':
            from .aae_trainer import AAETrainer
            trainer = AAETrainer
        elif ae_type == 'SAAE':
            from .saae_trainer import SAAETrainer
            trainer = SAAETrainer 
        elif ae_type == 'VAE_baseline':
            from .baseline_trainer import VAEBaseTrainer
            trainer = VAEBaseTrainer        
        return trainer


    def setup_metrics(self, metric_classes):
        self.metric_classes = metric_classes if isinstance(metric_classes, list) else [metric_classes]
        self.metric_names = [m.__name__ if m != 0 else 'mse_loss' for m in self.metric_classes]
        
        if self.metric_classes[0] != 0:
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
                    metric_on_train = metric_train(**kwargs)
                    self.metric_train[index].append(metric_on_train)

                metric_test = self.pack_for_metric(metric_class=metric, split='test', **kwargs)
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
            self.last_best_file = ["" for metric in self.metric_names]

    def model_saving(self, avg_epoch_loss, epoch):
        if self.save_models:
            scores = [self.metric_on_test[metric] if metric != "mse_loss" else avg_epoch_loss for metric in self.metric_names]
            for i, metric in enumerate(self.metric_names):
                if self.comparators[i](self.best_scores[i], scores[i]) == scores[i]:
                    self.best_scores[i] = scores[i]
                    
                    if self.last_best_file[i] != "":
                        os.remove(self.last_best_file[i])

                    torch.save(self.model.state_dict(),
                               os.path.join(self.log_dir, metric,
                                            self.__class__.__name__ + f"epoch{epoch}.pth"))
                    self.last_best_file[i] = os.path.join(self.log_dir, metric, 
                                                          self.__class__.__name__ + f"epoch{epoch}.pth")
               
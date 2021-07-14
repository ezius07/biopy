from datetime import datetime
import os
import numpy as np
from .trainer import Trainer
from .joint_trainer import JointTrainer
from .baseline_trainer import VAEPzEstimationTrainer, VAETrainer
from .distribution_trainer import VAETrainerDistribution
from .joint_double_discr import JointDoubleDiscr
from ..utils import HParams
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter


class ThanosTrainer:
    '''A small price to pay for salvation'''
    strategies_available = {"joint_estimator": {'agents': [JointTrainer],
                                                'stages': [2],
                                                'supports_tboard': True},

                            "one_shot": {'agents': [Trainer],
                                         'stages': [1],
                                         'supports_tboard': False},

                            "baseline": {'agents': [VAEPzEstimationTrainer, VAETrainer],
                                         'stages': [1, 1]},
                            
                            "baseline_1stage": {'agents': [VAETrainer],
                                                'stages': [1]},
                            
                            "distribution": {'agents': [VAEPzEstimationTrainer, VAETrainerDistribution],
                                             'stages': [1, 1]},

                            "distribution_1stage": {'agents': [VAETrainerDistribution],
                                                    'stages': [1]},

                            "joint_double_discr": {'agents': [JointDoubleDiscr],
                                                    'stages': [1]},
                            
                            }

    def __init__(self, par_dict, strategy=None, log_dir=None):
        self.__parameter_dict = {}
        self.results = []
        self.trainers = []
        self.dataset_args = {}
        self.model_args = {}
        self.train_args = {}
        self.parameter_dict = par_dict
        self.__strategy = None
        self.__state = None
        self.__init_state()
        if strategy is not None:
            self.strategy = strategy
        self.log_dir = log_dir if log_dir is not None else 'log_dir'

    def __init_state(self):
        self.__state = {'ALIVE'}

    def __update_state(self, method_name, agent):
        def updatefor_generate_dataset_loaders(agent):
            self.__state.add('DATA' + str(agent))
            self.__state = self.__state.difference({'DATA_DONE' + str(agent)})

        def updatefor_generate_models_optimizers(agent):
            self.__state.add('MODEL' + str(agent))
            self.__state = self.__state.difference({'MODEL_DONE' + str(agent)})

        def updatefor_train_model(agent):
            self.__state.add('TRAIN' + str(agent))
            self.__state = self.__state.difference({'TRAIN_DONE' + str(agent)})

        def updatefor_exec(agent):
            if 'DATA' + str(agent) in self.__state:
                self.__state = self.__state.union({'DATA_DONE' + str(agent)})
                self.__state = self.__state.difference({'DATA' + str(agent)})

            self.__state = self.__state.difference({'MODEL' + str(agent)}) \
                .union({'MODEL_DONE' + str(agent)})
            if 'TRAIN' + str(agent) in self.__state:
                self.__state = self.__state.union({'TRAIN_DONE' + str(agent)})
                self.__state = self.__state.difference({'TRAIN' + str(agent)})

        updater = locals()['updatefor_' + method_name]
        updater(agent)

    def __validate_state(self, method_name, agent):
        def validatefor_train_model(agent):
            if len(self.__state.intersection({'MODEL' + str(agent), 'MODEL_DONE' + str(agent)})) == 0 \
                    or len(self.__state.intersection({'DATA' + str(agent), 'DATA_DONE' + str(agent)})) == 0:
                raise Exception(
                    "Before being able to train, you have to call :\n-'generate_dataset_loaders'\n-'generate_models_optimizers'")

        def validatefor_exec(agent):
            if agent == -1:
                if self.__state.issubset({'ALIVE', 'STRAT'}):
                    raise Exception("To call exec you have to at least declare models")
                if 'STRAT' not in self.__state:
                    raise Exception("To call exec you have to declare a strategy")
            else:
                if (len(self.__state.intersection({'MODEL' + str(agent), 'MODEL_DONE' + str(agent)})) == 0) \
                        and (('TRAIN' + str(agent) in self.__state) or ('DATA' + str(agent) in self.__state)):
                    raise Exception(
                        f"You asked to generate data or to train for agent n.{agent}, but models declaration is missing")
                if 'TRAIN' + str(agent) in self.__state:
                    for ag in range(1, agent):
                        if len(self.__state.intersection({'TRAIN' + str(ag), 'TRAIN_DONE' + str(ag)})) == 0:
                            raise Exception(
                                f"You asked to train agent n.{agent}, but you didn't train agent n.{ag} first")

        validator = locals()['validatefor_' + method_name]
        validator(agent)

    def __check_state(self, method_name, agent, step):
        def checkfor_exec(agent, step):
            if step == 'INSTANCE':
                if len(self.trainers) < agent and ('MODEL' + str(agent) in self.__state):
                    return True
            else:
                if step + str(agent) in self.__state:
                    return True
            return False

        checker = locals()['checkfor_' + method_name]
        return checker(agent, step)

    def generate_dataset_loaders(self, dataset_class, agent=1, **kwargs):
        self.dataset_args[agent] = {'dataset_class': dataset_class}
        self.dataset_args[agent].update(kwargs)
        self.__update_state(self.generate_dataset_loaders.__name__, agent)

    def generate_models_optimizers(self, model_class, agent=1, **kwargs):
        self.model_args[agent] = {'model_class': model_class}
        self.model_args[agent].update(kwargs)
        self.__update_state(self.generate_models_optimizers.__name__, agent)

    def train_model(self, agent=1, **train_args):
        self.__validate_state(self.train_model.__name__, agent)
        self.train_args[agent] = train_args
        self.__update_state(self.train_model.__name__, agent)

    def print_state(self):
        self.state = self.__state

    def exec(self):
        self.__validate_state(self.exec.__name__, -1)

        for i, agent in enumerate(self.strategy['agents']):
            self.__validate_state(self.exec.__name__, i + 1)

            if self.__check_state(self.exec.__name__, i + 1, 'INSTANCE'):
                trainer_class = agent.get_trainer_for_model(self.model_args[i + 1]['model_class'])
                self.trainers.append(trainer_class(self.__parameter_dict[i + 1], log_dir=self.log_dir))
                self.results.append(None)

            if self.__check_state(self.exec.__name__, i + 1, 'DATA'):
                self.trainers[i].generate_dataset_loaders(**self.dataset_args[i + 1])

            if self.__check_state(self.exec.__name__, i + 1, 'MODEL'):
                self.trainers[i].generate_models_optimizers(**self.model_args[i + 1])

            if self.__check_state(self.exec.__name__, i + 1, 'TRAIN'):
                n_stages = self.strategy['stages'][i]

                res_old = self.results[i - 1] if i > 0 else None
                for stage in range(1, n_stages + 1):
                    self.results[i] = self.trainers[i].train_model(stage=stage, result=res_old,
                                                                   **self.train_args[i + 1])

                if isinstance(self.__parameter_dict[i + 1], HParams):
                    self.__parameter_dict[i + 1].warn_if_unused_keys()
                else:
                    print("Warning: using dict-like object for hyperparameters instead of a HParams instance!")
            self.__update_state(self.exec.__name__, i + 1)

        if self.strategy.get('supports_tboard', True):
            self.log_trainer(self.__parameter_dict.values(), self.trainers, self.train_args.values())

    @property
    def parameter_dict(self):
        return self.__parameter_dict

    @parameter_dict.setter
    def parameter_dict(self, par_dict):
        if isinstance(par_dict, list):
            for agent, pars in enumerate(par_dict):
                self.__parameter_dict[agent + 1] = pars
        else:
            self.__parameter_dict[1] = par_dict
        for i, trainer in enumerate(self.trainers):
            trainer.parameters = self.__parameter_dict[i + 1]
            # trainer.generate_models_optimizers(**self.model_args[i+1])

    @property
    def strategy(self):
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy):
        if strategy not in list(self.strategies_available.keys()):
            raise ValueError("The strategy specified does not exist, or it has not been implemented yet")
        self.__strategy = self.strategies_available[strategy]
        self.__state.add('STRAT')

    def log_trainer(self, hparameters_all, trainers, trainers_args):

        experiment = datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, experiment))
        hparams_all, metrics_all = {}, {}

        log_container = {}

        for agent_idx, (hparameters, trainer, train_args) in enumerate(zip(hparameters_all, trainers, trainers_args)):
            log_container[agent_idx] = {}
            metric_presence = train_args.get('metric', 'no_metric')

            if not isinstance(trainer.train_loss, dict):
                stage = 0
                log_container[agent_idx][stage] = {}
                log_container[agent_idx][stage]['train_loss'] = trainer.train_loss
                log_container[agent_idx][stage]['test_loss'] = trainer.test_loss

                log_container[agent_idx][stage]['metric_names'] = trainer.metric_names
                if metric_presence != 'no_metric':
                    log_container[agent_idx][stage]['metric_train'] = trainer.metric_train
                    log_container[agent_idx][stage]['metric_test'] = trainer.metric_test

            else:
                for stage in range(len(trainer.train_loss)):
                    log_container[agent_idx][stage] = {}
                    log_container[agent_idx][stage]['train_loss'] = trainer.train_loss[stage]
                    log_container[agent_idx][stage]['test_loss'] = trainer.test_loss[stage]
                    
                    log_container[agent_idx][stage]['metric_names'] = trainer.metric_names[stage]
                    if metric_presence != 'no_metric':                
                        log_container[agent_idx][stage]['metric_train'] = trainer.metric_train[stage]
                        log_container[agent_idx][stage]['metric_test'] = trainer.metric_test[stage]


            n_stages = len(log_container[agent_idx]) 
            for stage in log_container[agent_idx]:
                metric_dict = {}

                if n_stages == 1:
                    prefix_scalar = f"Stage {agent_idx}"
                    prefix_hparam = f"STAGE_{agent_idx}_"
                else:
                    prefix_scalar = f"Stage {agent_idx} part {stage+1} of {n_stages}"
                    prefix_hparam = f"STAGE_{agent_idx}_part_{stage+1}_of_{n_stages}"


                # Log train reconstruction loss
                for epoch, loss in enumerate(log_container[agent_idx][stage]['train_loss'][0], 1):
                    if epoch == 1:
                        best = loss
                    if best > loss:
                        best = loss
                    writer.add_scalar(f"{prefix_scalar}/Train reconstruction loss", loss, epoch)
                metric_dict[f"{prefix_scalar}/Train reconstruction loss [best]"] = best

                # Log train discrimination loss
                for epoch, loss in enumerate(log_container[agent_idx][stage]['train_loss'][1], 1):
                    if epoch == 1:
                        best = loss
                    if best > loss:
                        best = loss
                    writer.add_scalar(f"{prefix_scalar}/Train discrimination loss", loss, epoch)
                metric_dict[f"{prefix_scalar}/Train discrimination loss [best]"] = best

                # Log test reconstruction loss
                for epoch, loss in enumerate(log_container[agent_idx][stage]['test_loss'], 1):
                    if epoch == 1:
                        best = loss
                    if best > loss:
                        best = loss
                    writer.add_scalar(f"{prefix_scalar}/Test reconstruction loss", loss, epoch)
                metric_dict[f"{prefix_scalar}/Test reconstruction loss [best]"] = best

                if metric_presence != 'no_metric':
                    for metric_name, metric_train, metric_test in zip(log_container[agent_idx][stage]['metric_names'], 
                                                                      log_container[agent_idx][stage]['metric_train'],
                                                                      log_container[agent_idx][stage]['metric_test']):
                        # Log train metric
                        for epoch, metric in enumerate(metric_train, 1):
                            if epoch == 1:
                                best = metric
                            if best < metric:
                                best = metric
                            writer.add_scalar(f"{prefix_scalar}/Train metric {metric_name}", metric, epoch)
                        metric_dict[f"{prefix_scalar}/Train metric {metric_name} [best]"] = best
                        # Log test metric
                        for epoch, metric in enumerate(metric_test, 1):
                            if epoch == 1:
                                best = metric
                            if best < metric:
                                best = metric
                            writer.add_scalar(f"{prefix_scalar}/Test metric {metric_name}", metric, epoch)
                        metric_dict[f"{prefix_scalar}/Test metric {metric_name} [best]"] = best

                # --- just to avoid big names below
                metric_names = log_container[agent_idx][stage]['metric_names']
                # ---

                metrics_all.update(metric_dict)

            hparam_dict = {f'{prefix_hparam}_{hparam}': hparam_value for hparam, hparam_value in hparameters.items()}
            hparam_dict[f'{prefix_hparam}_omics'] = "-".join(hparam_dict.get(f'{prefix_hparam}_omics', []))
            hparam_dict[f'{prefix_hparam}_metric'] = ", ".join(metric_names) if metric_presence != 'no_metric' else 'no_metric'
            
            if f'{prefix_hparam}_PREPROCESS_LOG1P' in hparam_dict:
                hparam_dict[f'{prefix_hparam}_PREPROCESS_LOG1P'] = ", ".join(hparam_dict[f'{prefix_hparam}_PREPROCESS_LOG1P'])
            if f'{prefix_hparam}_BETAS' in hparam_dict:
                hparam_dict[f'{prefix_hparam}_BETAS'] = ", ".join(str(hparam_dict[f'{prefix_hparam}_BETAS']))
            for k in hparam_dict.keys():
                if isinstance(hparam_dict[k], list):
                    hparam_dict[k] = ", ".join(map(str, hparam_dict[k]))

            hparams_all.update(hparam_dict)

        # hparams function must be called with the complete dictionary of all hparams of all stages
        # you cannot do hparams({'lr_stage1': 1}, {}) and then later hparams({'lr_stage2': .5}, {})
        # you must do hparams({'lr_stage1': 1, 'lr_stage2': .5}, {})
        exp, ssi, sei = hparams(hparams_all, metrics_all)

        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        for k, v in metrics_all.items():
            writer.add_scalar(k, v)

        writer.close()

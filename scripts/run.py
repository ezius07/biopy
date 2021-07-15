# TOREMOVE, temporary fix
import sys
import os
import socket
# makes the biopy package visible no matter where the scripts
# are launched from - as long as you keep 'scripts' as a sibling of 'biopy'
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

# only for my local machine - gabT - TOREMOVE
hostname = socket.gethostname()
if hostname == "ga1i13o": 
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import os
from configparser import ConfigParser

from torch import nn
from biopy import metrics, datasets, models, training
from biopy.training import ThanosTrainer
from biopy.utils import HParams
import ast


def get_metric(name):
    if hasattr(metrics, name):
        return getattr(metrics, name)
    else:
        raise ValueError("'{}' is not a valid metric".format(name))


def get_loss(name):
    if hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise ValueError("'{}' is not a valid loss".format(name))


def get_dataset(name):
    if hasattr(datasets, name):
        return getattr(datasets, name)
    else:
        raise ValueError("'{}' is not a valid dataset".format(name))


def get_model(name):
    if hasattr(models, name):
        return getattr(models, name)
    else:
        raise ValueError("'{}' is not a valid model".format(name))


def get_model_cd4_closure(image_model, rna_model):
    image_model = get_model(image_model)
    rna_model = get_model(rna_model)

    def get_model_cd4(data_size, **kwargs):
        # Images
        # data_size è 1 perché lui controlla .shape[1] ma le immagini sono tensori 3d (c=1, h=64, w=64)
        if data_size == 1:
            return image_model(data_size=data_size, **kwargs)
        # RNA
        kwargs['n_hidden'] = 1024
        return rna_model(data_size=data_size, **kwargs)

    return get_model_cd4


class ParameterParser:
    """Class which allows reading tuning parameters from a .ini file"""

    def __init__(self, path, strategy, assert_sections=("GLOBAL", "OPTIMIZER", "SCHEDULER", "PREPROCESS", "LOSS",
                                                        "DATASET", "METRIC", "MODEL", "MODELCLASS")):
        """
        fold, log_dir, startegy as CLI
        all assert sections are one for each agent of the strategy e.g. DATASET_1, DATASET_2...
        GLOBAL: log_frequency, save_models
        OPTIMIZER: batch_size, lr, weight_decay et cetera
        SCHEDULER: step_size, gamma
        PREPROCESS: preprocess, preprocess_smote
        LOSS: discriminative_weight, reconstruction_weight ...
        DATASET: dataset object, omics_dataset
        METRIC: metric object, mean_strategy, eval_freq
        MODEL: hidden_size, omics_model
        MODELCLASS: one for each agent of the strategy (MODEL_1, MODEL_2...) -> model_class, discriminator_class ...

        Default constructor
            Input:  - path .ini file
                    - sections to check to be present in the .ini file
            Output: //
        """

        try:
            num_agents = len(training.ThanosTrainer.strategies_available[strategy]["stages"])
        except KeyError:
            raise Exception("Strategy passed {} is not available".format(strategy))

        self.__config = ConfigParser()
        self.__config.optionxform = str
        self.__config.read(path)
        self.__assert_sections = assert_sections

        self._params = [{"HPARAMS": {}, "MODELCLASS": {}, "METRIC": {}, "DATASET": {}} for _ in range(num_agents)]

        for agent in range(num_agents):
            agent += 1
            for section in self.__assert_sections:
                section = section + "_" + str(agent)
                try:
                    self.__config[section]
                except KeyError:
                    raise Exception('Section {} is not present in .ini file".format(section)'.format(section))

        for section in self.__config:
            for param in self.__config[section]:
                el_value = ast.literal_eval(self.__config[section][param])
                agent = section.split("_")[-1]
                if section.split("_")[-2] not in ["DATASET", "METRIC", "MODELCLASS"]:
                    if param == "reconstruct_loss":
                        el_value = get_loss(el_value)()
                        self._params[int(agent) - 1]["METRIC"][param] = el_value
                        continue
                    if param == "omics":
                        if type(el_value) != list:
                            el_value = [el_value]
                    self._params[int(agent) - 1]["HPARAMS"][param] = el_value
                else:
                    if section.split("_")[-2] == "DATASET":
                        if param.split("_")[-1] == "class":
                            el_value = get_dataset(el_value)
                        if param == "omics":
                            if type(el_value) != list:
                                el_value = [el_value]
                    if section.split("_")[-2] == "MODELCLASS":
                        if param.split("_")[-1] == "class":
                            if self.__config["DATASET_" + agent]["dataset_class"] == "DatasetMultiOmicsNatureTrainTest" \
                                    and len(self.__config["MODEL_" + agent]["omics"]) == 2:
                                el_value = get_model_cd4_closure(self.__config["MODELCLASS_" + agent]['image_model'],
                                                                 self.__config["MODELCLASS_" + agent]['rna_model'])
                            else:
                                el_value = get_model(el_value)
                    if section.split("_")[-2] == "METRIC":
                        if param.split("_")[-1] == "class":
                            metrics_classes = []
                            for el in el_value:
                                metrics_classes.append(get_metric(el))
                            el_value = metrics_classes
                            self._params[int(agent) - 1][section.split("_")[-2]]["metric"] = el_value
                            continue
                    self._params[int(agent) - 1][section.split("_")[-2]][param] = el_value

    def get_param(self, agent, section, name):
        return self._params[agent - 1][section].get(name, None)

    def get_section(self, agent, section):
        return self._params[agent - 1][section]


def main(config_path, fold, log_dir, strategy):
    if os.path.exists(log_dir):
        raise Exception('log folder {} already exists!'.format(log_dir))

    os.makedirs(log_dir)

    try:
        num_agents = len(training.ThanosTrainer.strategies_available[strategy]["stages"])
    except KeyError:
        raise Exception("Strategy passed {} is not available".format(strategy))

    params = ParameterParser(config_path, strategy)

    parameter_dict = [HParams(**params.get_section(agent, "HPARAMS")) for agent in range(num_agents)]

    tt = ThanosTrainer(parameter_dict, log_dir=log_dir)
    tt.strategy = strategy

    for agent in range(num_agents):
        agent += 1
        kwargs_dataset = params.get_section(agent, "DATASET")
        kwargs_dataset = {k: v for k, v in kwargs_dataset.items() if v is not None}
        tt.generate_dataset_loaders(agent=agent, folder=fold, **kwargs_dataset)

        kwargs_model = params.get_section(agent, "MODELCLASS")
        kwargs_model = {k: v for k, v in kwargs_model.items() if v is not None}
        tt.generate_models_optimizers(agent=agent, **kwargs_model)

        kwargs_metric = params.get_section(agent, "METRIC")
        kwargs_metric = {k: v for k, v in kwargs_metric.items() if v is not None}
        tt.train_model(agent=agent, **kwargs_metric)

    tt.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, help="dataset path")
    parser.add_argument("--log_dir", required=True, help="directory to store the log files")
    parser.add_argument("--strategy", required=True, help="strategy to perform the training")
    parser.add_argument("--config_path", required=True, help="path of the configuration file")
    args = parser.parse_args()

    main(args.config_path, args.fold, args.log_dir, args.strategy)

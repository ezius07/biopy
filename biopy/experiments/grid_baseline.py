import argparse

from ..training import ThanosTrainer
from ..datasets import DatasetMultiOmicsGDCTrainTest
from ..models import FC_VAE, Simple_Classifier, Adversarial_Classifier
from ..metrics import ROCRandomForest, OmicsReconstructionError
from ..utils import HParams

import json
from sklearn.model_selection import ParameterGrid

def pprint_dict(d):
    print(json.dumps(d, indent=4, sort_keys=True))

grid_stage1_ = {
    # Optimizer
    'BATCH_SIZE': [32],
    'LR': [1e-3, 1e-4],
    'LR_D': [1e-3],
    'STEP_SIZE': [100],
    'GAMMA': [0.5],
    'WEIGHT_DECAY': [0],
    'WEIGHT_DECAY_D': [0],
    'NUM_EPOCHS': [50],
    
    # Loss
    'BETA_KLD': [1e-3, 1e-5, 1e-7],
    'BETA_DISCR': [1e-1, 1e-3, 1e-5],

    # Model
    'HIDDEN_SIZE': [128],

    # Dataset
    'PREPROCESS': ['minmax_fs_std'],
    'PREPROCESS_FS_TOP_N': [1000],
    'LABELS_COLUMNS_': [['sample_type']],
    'OMIC_': ['mRNA'],
    'PREPROCESS_SMOTE': [True],
    'SAVE_MODELS': [False]
}
grid_stage2_ = {
    # Optimizer
    'BATCH_SIZE': [32],
    'LR': [1e-3, 1e-4],
    'LR_D': [1e-3],
    'STEP_SIZE': [100],
    'GAMMA': [0.5],
    'WEIGHT_DECAY': [0],
    'WEIGHT_DECAY_D': [0],
    'NUM_EPOCHS': [50],

    # Loss
    'BETA_KLD': [1e-3, 1e-5, 1e-7],
    'BETA_DISCR': [1e-1, 1e-3, 1e-5],
    'ALPHA_DISCR': [1],
    'SAVE_MODELS': [False],
    
    # Model and Dataset: derived from stage1
    'PREPROCESS_LOG1P': [['miRNA']]
}

def run(dataset_folder, log_frequency, log_dir):

    grid_stage1 = list(ParameterGrid(grid_stage1_))
    grid_stage2 = list(ParameterGrid(grid_stage2_))

    training_i = 1
    training_count = len(grid_stage1) * len(grid_stage2)

    print('Number of combinations:', training_count)

    tt = ThanosTrainer(par_dict=[])
    tt.strategy = 'baseline'
    tt.log_dir = log_dir
    for i, hparams_stage1 in enumerate(grid_stage1):
        
        omic_stage1 = hparams_stage1.pop('OMIC_')
        labels_stage1 = hparams_stage1.pop('LABELS_COLUMNS_')
        # N_LABELS represents the total number of distinct values that the label column has
        # Since in our case, both sample_type and project_id have got 2 distinct values each,
        # the total number of distict values is 4 if both are selected and 2 if only one is selected
        hparams_stage1['N_LABELS'] = 2 * len(labels_stage1)
        tt.generate_dataset_loaders(DatasetMultiOmicsGDCTrainTest, agent=1, folder=dataset_folder, 
                                omics=[omic_stage1], labels_columns=labels_stage1)
        tt.generate_models_optimizers(agent=1, model_class=FC_VAE, discriminator_class=Simple_Classifier)
        tt.train_model(agent=1) # Training is not performed here

        batch_size_stage2 = None
        for j, hparams_stage2 in enumerate(grid_stage2):
            # The dataset in the second stage must be preprocessed the same way as in the first stage,
            # the only difference is that we take all omics
            # Since it is always preprocessed in the same way for all the second stages,
            # we load the dataset only once
            # However we also need to reload the dataset if the batch size has changed in order to update
            # also the dataloaders' batch size parameter
            all_omics = ["miRNA", "mRNA", "meth27-450-preprocessed"]
            if batch_size_stage2 != hparams_stage2['BATCH_SIZE']:
                tt.generate_dataset_loaders(DatasetMultiOmicsGDCTrainTest, agent=2, folder=dataset_folder, 
                                    omics=all_omics, labels_columns=labels_stage1)
                batch_size_stage2 = hparams_stage2['BATCH_SIZE']

            hparams_stage2['HIDDEN_SIZE'] = hparams_stage1['HIDDEN_SIZE']
            if 'PREPROCESS' in hparams_stage1:
                hparams_stage2['PREPROCESS'] = hparams_stage1['PREPROCESS']
            if 'PREPROCESS_FS_TOP_N' in hparams_stage1:
                hparams_stage2['PREPROCESS_FS_TOP_N'] = hparams_stage1['PREPROCESS_FS_TOP_N']
            if 'PREPROCESS_LOG1P' in hparams_stage1 and 'PREPROCESS_LOG1P' not in hparams_stage2:
                hparams_stage2['PREPROCESS_LOG1P'] = hparams_stage1['PREPROCESS_LOG1P']
            if 'PREPROCESS_SMOTE' in hparams_stage1 and 'PREPROCESS_SMOTE' not in hparams_stage2:
                hparams_stage2['PREPROCESS_SMOTE'] = hparams_stage1['PREPROCESS_SMOTE']
            if i == 0 and j == 0:
                # Read by JointTrainer for instantiating the correct number of VAEs when
                # the JointTrainer itself is instantiated
                # Since the JointTrainer is instantiated only once, set this parameter
                # only the first time (otherwise HParams will RIGHTFULLY complain that
                # we are setting an hyperparameter but not reading it)
                hparams_stage2['omics'] = all_omics

            hparams_stage1['LOG_FREQUENCY'] = hparams_stage2['LOG_FREQUENCY'] = log_frequency

            tt.generate_models_optimizers(agent=2, model_class=FC_VAE, discriminator_class=Adversarial_Classifier)
            tt.train_model(agent=2, metric=[ROCRandomForest, OmicsReconstructionError], mean_strategy='only_translations') 

            # Setting hyperparameters in the form [hparams_stage1, hparams_stage2]
            # Please note that, even if we are always setting the hyperparameters for all stages,
            # we are only training the second stage if the first stage has not been trained.
            tt.parameter_dict = [HParams(**hparams_stage1), HParams(**hparams_stage2)]

            print()
            print(' --- Starting a new training ---')
            pprint_dict(hparams_stage1)
            pprint_dict(hparams_stage2)
            print('Training {}/{} is being performed'.format(training_i, training_count))

            try:
                tt.exec() # Actual training of stage2 (and stage1 if needed) starts here
            except ValueError as e:
                if e.args[0] == "Input contains NaN, infinity or a value too large for dtype('float32').":
                    print("Input contains NaN, infinity or a value too large for dtype('float32').")
                    print("Training is stopped here and logged")
                    tt.log_trainer(tt.parameter_dict.values(), tt.trainers, tt.train_args.values())
                else:
                    raise e
            training_i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset-folder', type=str, required=True)
    parser.add_argument('-lf', '--log-frequency', type=int, default=1)
    parser.add_argument('-ld', '--log-dir', type=str, default='log_baseline_grid')
    run(**vars(parser.parse_args()))
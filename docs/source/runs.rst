********************
Replicating results
********************


After having presented the technical details of how the different pipelines are implemented, this section 
provides a more practical approach with snippets on how to run experiments to replicate our results.


.. warning::
    To execute the snippets in this section you need to install our package and its dependencies.
    See :ref:`Installation` for details on how to do that and download the datasets as well.

Strategies 
================


We provide a ready-made script to execute all the proposed implementations. Below more details on the parameters
that you need to specify, but the main one are 

* the dataset
* the training method aka the strategy

For a more specific description of how are strategy used inside our framework see :ref:`The trainer wrapper`.

For a more theorical perspective on said strategies, see instead our presentation.

Here we report just a brief summary of all the available strategies that you can ask our framework tu run:

* **one_shot** 
    Most basic implementation. Allows to train a standard VAE, AAE, or Supervised AAE on a single domain

* **baseline**
    Implements the  method reported as baseline, with a first stage (meant for image domains) with a discriminator 
    on the labels. The second stage has the discriminator as well and uses an anchor loss when available, on paired datasets.

* **baseline_1stage**
    Meant for datasets that do not include an image domain. Only executes the second stage of 
    the baseline explained above

* **joint_estimator**
    Our first proposed variation; to avoid the need for an anchor loss to match the latent space, tries 
    to estimate in a first stage a distribution conditioned on the label; and then to impose that distribution
    in a second stage separately per each domain/omic, using the framework of Adversarial AEs.

* **joint_double_discr**
    Same principle of the previous one, but instead of using AAEs uses a discriminator to impose 
    a common distribution.

* **distribution**
    Apply principles from domain adaptation to ensure that the different domains get encoded on a common
    distribution. Possible to specify a weighted combination of MMD, HAFN, SAFN losses.
    Includes a first stage of pretraining, meant for image domains.

* **distribution_1stage**
    Same as above but without the first stage of pretraining


Run the script 
================

A script to easily replicate all the reported results is provided at `scripts/run.py`.
It does not add anything to the proposed framework, and it is provided for convenience of the external user
to take care of the argument parsing and config specification.

The high level parameters are passed to the command line, such as the dataset path, the strategy, logging 
folder. Instead for the more specific details, a config file is needed. Config files ae represented as `.ini` files
with different sections, one for each of the different aspects to configure.
All the config files for the reported experiments are provided under `configs`, ready for you to try them out.

Here is a summary of the command line argument:

.. code:: console

    usage: run.py [-h] --fold FOLD --log_dir LOG_DIR --strategy STRATEGY --config_path CONFIG_PATH

    arguments:
    -h, --help                  show this help message and exit
    --fold FOLD                 dataset path
    --log_dir LOG_DIR           directory to store the log files
    --strategy STRATEGY         strategy to perform the training
    --config_path CONFIG_PATH   path of the configuration file


In our `configs` folder, you find more files than the ones mentioned in the snippet below.
For the usage of any of them, the filename is very explicative, containing the dataset, the methodology, and any additional 
infromation such as additional losses (e.g. `kld` ) and preprocessing techniques (e.g. `smote`)

Reproducing baseline results
-----------------------------

Below we report the command to reproduce the results applying the baseline method and models to all the 3 datasets.

* **CD4** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/cd4_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline \
                --config_path configs/CD4/cd4_baseline.ini

* **A549** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/a549_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline_1stage \
                --config_path configs/A549/a549_baseline.ini

* **GDC** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/gdc_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline_1stage \
                --config_path configs/GDC/gdc_baseline.ini


Joint Training with Adversarial AEs and double discriminator
-------------------------------------------------------------

This snippets are to run experiments with our proposed variation - 2 stage Joint Training, with a double
 discriminator.


* **CD4** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/cd4_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline \
                --config_path configs/CD4/cd4_baseline.ini

* **A549** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/a549_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline_1stage \
                --config_path configs/A549/a549_baseline.ini

* **GDC** :
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/gdc_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy baseline_1stage \
                --config_path configs/GDC/gdc_baseline.ini


Training with losses from Domain adaptation:
---------------------------------------------

* **CD4** and **MMD**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/cd4_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution \
                --config_path configs/CD4/cd4_mmd_kld_no_smote.ini

* **CD4** and **HAFN**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/cd4_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution \
                --config_path configs/CD4/cd4_hafn_no_smote.ini

* **A549** and **MMD**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/a549_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution_1stage \
                --config_path configs/A549/a549_mmd.ini


* **A549** and **SAFN**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/a549_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution_1stage \
                --config_path configs/A549/a549_safn.ini

* **GDC** and **MMD**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/gdc_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution_1stage \
                --config_path configs/GDC/gdc_mmd_no_smote.ini

* **GDC** and **HAFN**:
    .. code:: console
        
        python scripts/run.py --fold /your/path/to/gdc_folder \
                --log_dir /where/you/will/find/logs \ 
                --strategy distribution_1stage \
                --config_path configs/GDC/gdc_hafn_no_smote.ini



Plot latent spaces
===================

It may be useful to take a look inside your model and plot what is happening inside your latent space.
We provide a utility to do conveniently do so. Take a dataset, a model (hopefully trained previously),
and the function will plot the different domains in the dataset, choosing a main color to each label, 
and then assign a gradient of that color to each domain.
Ideally if the domain translation has been successfull, you should see that different gradients of each color 
(e.g. samples from different domains and same label) are overlapped in a common distribution.

Below we report examples for all the datasets, calling the `main` function provided in `scripts/visualizer.py`

* **CD4**
    
    .. code-block:: py3

        from visualizer import main
        main(dataset_name='cd4', dataset_folder='dataset_nature',
            model_name='cd4_vae',
            checkpoints="results/cd4/mmd/ROCCNNRF/{omic}VAETrainerDistributionepoch891.pth", 
            output_path='cd4.png')


* **A549**
    
    .. code-block:: py3

        from visualizer import main
        main(dataset_name='a549', dataset_folder='dataset_nature_atac-rna',
            model_name='a549_vae',
            checkpoints="results/a549/baseline/KNNAccuracySklearn/{omic}VAETrainerepoch1494.pth", 
            output_path='cd4.png')


* **GDC**
    
    .. code-block:: py3

        from visualizer import main
        main(dataset_name='gdc', dataset_folder='dataset_breast',
            model_name='gdc_vae',
            checkpoints="results/gdc/gdc_mmd/ROCCNNRF/{omic}VAETrainerDistributionepoch103.pth", 
            output_path='gdc.png')
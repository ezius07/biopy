********************
Replicating results
********************


After having presented the technical details of how the different pipelines are implemented, this section 
provides a more practical approach with snippets on how to run experiments to replicate our results.

See :ref:`Installation` for details on how to download the datasets and the source code.

Run the script 
================

A script to easily replciate all the reported results is provided at `scripts/run.py`.
It does not add anything to the proposed framework, and it is provided for convenience of the external user
to take care of the argument parsing and config specification.

The high level parameters are passed to the command line, such as the dataset path, the strategy, logging 
folder. Instead for the more specific details, a config file is needed. All the config files for the reported 
experiments are provided under `configs` folder in the format of `.ini` files.

Here is a summary of the command line argument:

.. code:: console

    usage: run.py [-h] --fold FOLD --log_dir LOG_DIR --strategy STRATEGY --config_path CONFIG_PATH

    arguments:
    -h, --help                  show this help message and exit
    --fold FOLD                 dataset path
    --log_dir LOG_DIR           directory to store the log files
    --strategy STRATEGY         strategy to perform the training
    --config_path CONFIG_PATH   path of the configuration file

Baseline 
----------------------

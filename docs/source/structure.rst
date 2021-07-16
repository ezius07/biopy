*********************
Repository structure
*********************

In this section the structure of the repository is explained. The tree-like 
representation is reported below:

::

    repository_root
    |
    ├── configs
    │   ├── A549
    │   ├── CD4
    │   └── GDC
    |
    ├── dependencies.txt
    ├── install
    ├── install.bat
    ├── LICENSE
    ├── pyproject.toml
    ├── README.md
    |
    ├── scripts
    │   ├── download_dataset.sh
    │   └── run.py
    |
    ├── setup.cfg
    |
    ├── src
       └── biopy
           ├── __init__.py
           ├── datasets
           ├── experiments
           ├── metrics
           ├── models
           ├── statistic
           ├── training
           └── utils


Our implementation is released as a python package. The source code is in the 
folder `src/biopy`. The 'scripts' folder contains instead examples and code that 
use the tools inside the package. Their purpose is to reproduce the result that we reported.

The package is made up of sub-packages each containing the tools to tackle different aspects of the proposed implementation.

* 'datasets' 
    Everything concerning the handling of the different datasets that were employed
    in the various tasks. Due to the highly variable formats of GDC datasets with respect to the Nature datasets,
    there are different classes for each of those, although they do inherit from a custom class that was 
    written to provide the general functionalities of preprocessing and train-val-test splitting. All the classes
    also inherit from the PyTorch Dataset class in order to fully exploit the framework.

* 'experiments' 
    This sub-packages is related to the fine-tuning of the different architectures and contains
    the code to perform a grid-search on a given model and set of hyper-parameters.

* 'metrics' 
    In tasks such as Multi-domain translation it is often hard to have a good evaluation of how well
    models are actually performing. This sub-package provides a number of metrics for that purpose. Some of them
    focus more on the latent space structure (KNN accuracy, Fraction Closer) others on the domain translation
    (Reconstruction error, classifier on the translations) 

* 'models'  
    Contains the implementations of all the architectures used. Reported as Baseline are the models 
    implemented to reproduce results of the paper taken as starting point. The other are custom networks, divided 
    between convolutional and non - based on whether or not they were thought for image data.
    
* 'statistic' 
    This package is used when training Adversarial AE that need a distribution against which match their
    latent spaces. The classes in here provide a way to either specify a distribution in terms of combination of multi-dimensional
    Gaussians or Laplace distribution, or provide a model and have a sampler generated from the dataset encoded by 
    the given model. This package isolates the logic for sampling making the code easier to read and mantain.

* 'training' 
    In this package is implemented a multi-method training framework used to easily switch between training methods,
    datasets, models; simply  by changing a couple of parameters. The wrapper class, biopy.training.ThanosTrainer, provides the interface
    for the external user to specify the overall parameters, and then internally there are the specific classes each one implementing
    a different training method. The training classes all inherit from biopy.training.Trainer that provides base mechanisms such as
    model checkpoints, and access to metrics.

* 'utils' 
    Contains all kinds of utilities functions and classes of various nature; such as access to API for downloading dataset, formatting data,
    extract useful info from models and plot results. 
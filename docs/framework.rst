************************
Training framework
************************

This section covers the training framework that we implemented. In this page the approach
is more technical with the purpose of explaining the structure of the classes; so 
that in the next section about replicating results the usage can be clearer.

The reason why we needed to structure such a framework was to provide an easy way to switch
among different training methods and datasets without having to rewrite a pipeline from scratch every time.
So the basic idea is to devise an interface that represents a pipeline, and then have many classes implementing
it each one for a different purpose; so that then it can be easy for an external wrapper 
to just call the same methods on different classes.

Below we briefly explain the class diagram, and then the pipeline interface implemented by all of them
is reported more in depth.

.. warning::
    Although we are calling it an 'interface', it is not actually an interface in the strict
    technical OOP language, as in Python there is not the possibility of declaring (at least not easily) interfaces,
    and so we just have many classes that implement the same methods

Classes diagram
========================

From now on we will refer to all the classes that implement the pipeline-representing interface as 'trainer classes'.
The base class from which they all inherit is `biopy.training.Trainer`.

biopy.training.Trainer 
----------------------

This class represents the most general case of training on a single domain, and so in its methods it instantiates a single
dataset and a single model. It actually does not implement a training loop, that is left to the sub-classes to specify.
Even though it is very general, it implements many methods that can be reused by all the trainer classes, such as for validation,
dataset preprocessing, and other general setups.

There are 2 groups of classes inheriting from Trainer:
 *  All the classes that implement training on a single domain. For example, `biopy.training.VAETrainer` , that implements 
    training of a single Adversarial AE. These kind of classes are very easy to implement, as they can inherit everything from Trainer, 
    the user only needs to specify the dataset and model classes. What these sub classes do have to implement is the training method, as that 
    is specific to every case.   
 * The `biopy.training.JointTrainer` class. The next section explains about that.

biopy.training.JointTrainer 
----------------------------

This class instead is the base for all the methods that train on multiple domains; and so require many dataset and model objects,
one per domain, or as they are identified in the code, one per 'omic'.
Due to this fundamental difference, it overrides almost all the methods from its parent Trainer, expect for the validation method and few others.

Unlike Trainer which is just a general template, JointTrainer does implement a training method; which is
the joint training on all domains to estimate a latent distribution, followed by a second stage where it imposes 
the estimated distribution. For more details and the motivations, refer to the presentation. 

From this class inherit all the trainer that implement a method involving more than one domain at once.

Other classes
----------------

Ultimately, the classes that implement our proposed approaches all have to deal with multi-domain setting, and so they 
inherit from JointTrainer. Examples are `src.training.VAETrainerDistribution`, `src.training.VAETrainer` and `src.training.JointDoubleDiscr`.
Thanks to this inheritance scheme, many of these classes only need to specify their training loop, and they get "for free" from the superclass
access to many useful structures; like per-omic dataset and model objects, logging and metric support.



Now with this scheme in mind we explain the methods through which these classes are able to implement a full pipeline.

Representing a pipeline
========================

Here is explained the way in which a pipeline is represented inside all the classes in the training subpackage.
List of methods and their purpose:

* `__init__(**kwargs)`

    Receives a dictionary with all the hyperparameters and other available customizations.

* `generate_dataset_loaders(dataset_class, **kwargs)`

    It has the job of loading the dataset and instantiating pytorch DataLoaders. In the parameters you specify
    which is the class to be instantiated, and the other `**kwargs` are forwarded to the constructor of the dataset.
    Each class can customize this method to its needs; for example classes that handle training of 
    all the omics at once can instantiate here the datasets for all of them and store it in a dictionary with 
    omic names as keys.
    The reason why the dataset class is left as parameter to specify is to give more freedom to test different classes
    during development. Once the code is 'deployed', that part can be further automatized by fixing the dataset class for 
    a given task. 

* `preprocess_dataset(dataset, parameters)`

    This method moves all the logic of the preprocessing in a single place. It takes care of parsing the parameters specified by the user, and
    applying them on the provided dataset object. The actual implementation of the preprocessing is in the dataset classes;
    and this is just an interface that ensures compatibility among the different classes.

* `generate_models_optimizers(model_class, optimizer_SGD, **kwargs)`

    The purpose of this method is to build the model and the other objects required for training (scheduler, optimizer)
    It receives as parameter the class that implements the model to be used, and it instantiates it forwarding the `**kwargs` to it.
    The `optimizer_SGD` is a boolean variable that allows to specify whether to use the SGD or the Adam optimizer.
    As for the dataset here each class has the freedom to set up its environment in the preferred way, instantiating a model
    for each omic, a shared or not discriminator, and so on. 

* `train_model(**kwargs)`

    This is of course the core method that each class has to implement.
    At this point the dataset are loaded and the models are built, so the job of this method is to
    implement the training loop specific for each method.
    It specify the way in which data are forwarded, which losses are used and so on.
    The other parameters can specify for example which metrics to test on or some custom parameters.

Other utilities methods:
The following methods are of general utility and are implemented only by Trainer and JointTrainer;
as the only variability in these comes from whether or not there are multiple domains to consider.
All the other classes get these methods "for free" and can use them with no need of overriding.

* `pack_for_metric(metric_class, split, **kwargs)`

    Just like the trainer classes, also the metrics are implemented with a common structure. They all receive the datasets on which to test and a model
    inside a dictionary; so this method acts as an interface between the trainer classes and the metrics.
    The trainer classes call this for all the metrics they need to evaluate, and this method takes care of 
    properly instatiating the metric class.
    The `split` parameter differentiates between evaluating the metrics on the train or test set.

* `validate_on(net, data_loader, criterion, device)`

    This method is meant for a simple validation pass on a given model and dataset.
    It is probably the most standard method across the different methods and so it is not overriden
    by anyone

* `setup_metrics(metric_classes)`

    To be called before starting the training. It receives the classes of the metrics to be used during
    training and sets up the data structure to collect results.

* `eval_metrics(**kwargs)`

    Called after every epoch, takes care of calling the metrics (for each domain in the case of JointTrainer)
    Moves in one place all the complexity regarding instantiating the metrics that the user specified to evaluate,
    calling them passing the proper parameters and collecting results.

* `setup_model_saving(save_models)`

    Called before starting the training loop, it receives the parameter specified by the user that asks whether or not
    to save the best models. It sets up the directory in which to do so, and data structures to hold the
    best results. It saves the best models for each of the metrics that were asked to evaluate. 

* `model_saving(avg_loss, epoch)`

    Called after every epoch receiving the epoch number (for logging purposes) and the average loss that is used
    as criteria to find the best model if no metrics were specified. It removes the file for the
    previous best model and saves a new one, putting in the name the metric, the domain and the epoch number.


The trainer wrapper
============================

On top of the trainer classes described above, there is a wrapper to provide easy and structured
access to all the training subclasses; this class is `ThanosTrainer` in `biopy.training.trainer_wrapper.py`
It offers a way to specify the parameters for each step of the training pipeline.

Most of the code in the class is boilerplate that makes internal state checks to ensure that its methods
are called in the right order to build the pipeline, providing a robust interface to easily switch
among training methods and datasets specifying all the needed hyperparameters.

Training methods are represented in the form of "strategies", that are encoded in dictionaries inside this class.
The user has to set the name of the strategy, and the class automatically knows how many trainer class are needed to perform
that strategy, and how to pipeline them.
Each strategy can have one or more agents; where an agent is a trainer class, and each agent can implement one or more stages

Beyond setting the strategy, what the user has to do is to call, for each agent class that composes the strategy,
the main methods that represent its pipeline (as described above) in order to specify the required parameters.
Specifically these methods are :

* `generate_dataset_loaders(dataset_class, **kwargs)`

* `preprocess_dataset(dataset, parameters)`

* `generate_models_optimizers(model_class, optimizer_SGD, **kwargs)`

* `train_model(**kwargs)`

Upon call of these methods Thanos will not perform any action, as these calls represent the statement of
the parameters of the pipeline that has to be followed. After this step, when the user wants to start
the training, it is possible to do so by calling :

* `exec()`

That will start the pipeline relative to the strategy that has been set.
Since the trainer classes require in many cases a number of other parameters that are specified in a dictionary upon initialization; in order
to provide these parameters it is possible to pass them to the Thanos constructor, in the form of a list with 
an item for each agent class of the strategy.
This is useful to specify many useful behaviors such as ask to save models, specify log directory and so on.

Now, of course beyond running experiments, what one is typically interested in is looking at results.
Another useful feature offered by `ThanosTrainer` is logging, through the tensorboard platform.
After every run, it will log first of all, all the hyperparameters used; and the progress of training.
The latter by default is composed of the training and validation loss; additionally, if any metrics were specified to 
be evaluated, they will be logged as well.

If asked to do so, the best performing models for each metric will be saved too.

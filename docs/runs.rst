********************
Replicating results
********************


The main class that is a wrapper to provide access to all the training subclasses is 
`ThanosTrainer` in `biopy.training.trainer_wrapper.py`

The trainer wrapper
============================

The `ThanosTrainer` class provides an easy way to run all the different training methods on all the
available datasets, allowing to replicate our results.
It offers a way to specify the parameters for each step of the training pipeline.

Most of the code in the class is boilerplate that makes internal state checks to ensure that its methods
are called in the right order to build the pipeline.

Training methods are represented in the form of "strategies", that are encoded in dictionaries inside this class.
The user has to set the name of the strategy, and the class knows how to implement it.
Each strategy can have one or more agents; where an agent is a class, and each agent can implement one or more stages

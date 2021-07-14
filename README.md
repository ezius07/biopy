# Multiomics_autoencoders_BioProject

## How to download the GDC dataset

If you want to recreate the dataset from the GDC portal you can simply run from bash the following command:
`./download_dataset.sh`

If you want to run the script step by step, for each action (`download_omic`, ...), there are additional options for customizing directories' locations and other relevant parameters.
See details with `python3 download_dataset_gdc.py {action} -h` or `python3 download_dataset_gdc.py -h` to get the list of available actions

otherwise we have temporarly uploaded the preprocessed and splitted files on a server, so you can download in a faster way the data with the following command:
`wget -r -nH --reject="index.html*" --no-parent http://lino1.francesco.pw/bio/dataset_breast/ && mv bio/* .`

## How to use the dataset classes
The `DatasetMultiOmics` class can be used to read the synthetic dataset, while `DatasetMultiOmicsGDC` and `DatasetMultiOmicsGDCTrainTest` can be used to read the GDC dataset downloaded with the aforementioned tool. These classes expect files to be in tsv format with .txt extension and samples along the columns.

### Basic synthetic dataset usage
In the basic usage scenario, you load all omics and then use one at a time, for instance:

```
from datasets import DatasetMultiOmics

dataset = DatasetMultiOmics(folder='dataset_5000samples', omics=('mRNA', 'meth',), labels='clusters')

dataset_train, dataset_test = dataset.train_val_test_split(test=.25)

mean, std = dataset_train.standardize(all_omics=True)
dataset_test.standardize(all_omics=True, mean=mean, std=std)

dataset_train.set_omic('mRNA')
model1 = train(dataset)
evaluate(dataset_test)

dataset_train.set_omic('meth')
model2 = train(dataset)
evaluate(dataset_test)
```

### Basic GDC dataset usage
In order to use the downloaded and processed GDC dataset, `DatasetMultiOmicsGDC` and `DatasetMultiOmicsGDCUnbalanced` classes can be used. These two classes differ from `DatasetMultiOmics` because the label can be programmatically chosen since the downloaded label file is richer amd contains many columns. E.g. For a given experiment, one may want to use the `sample_type` column as a label, but for another experiment the label can be derived from the `project_id`; multiple columns can also be used.

Furthermore:
 - `DatasetMultiOmicsGDC` has to be used when all samples have all the omics and all samples share the same label file, as it happens for the synthetic dataset;
 - `DatasetMultiOmicsGDCUnbalanced` has to be used when instead not all samples have all omics (for each omic there is a separate label file).

E.g. for using the `DatasetMultiOmicsGDCUnbalanced` class with the downloaded GDC dataset, which is "unbalanced" as intended above:

```
from datasets import DatasetMultiOmicsGDCUnbalanced

# The label here is sample type (2 possible values) + project id (2 possible values), so 4 possible values
dataset = DatasetMultiOmicsGDCUnbalanced(folder='dataset', omics=('mRNA', 'miRNA',), labels_columns=('sample_type', 'project_id'))

dataset.standardize(all_omics=True)

dataset.set_omic('mRNA')
model1 = train(dataset)

dataset.set_omic('miRNA')
model2 = train(dataset)
```

One shortcoming of the `DatasetMultiOmicsGDCUnbalanced` class is that the `train_val_test_split` method is not available.
However the download script provides a command to split the downloaded dataset in such a way that the test set is made of samples that have all the three omics.
To load it:

```
from datasets import DatasetMultiOmicsGDCUnbalanced

dataset_train = DatasetMultiOmicsGDCUnbalanced(folder='dataset', omics=('train_mRNA', 'train_miRNA',))
dataset_test = DatasetMultiOmicsGDC(folder='dataset', omics=('test_mRNA', 'test_miRNA',))

mean, std = dataset_train.standardize(all_omics=True)
dataset_test.standardize(all_omics=True, mean=mean, std=std)
```

The code above is equivalent to the simplified code below which uses the class `DatasetMultiOmicsGDCTrainTest`:

```
from datasets import DatasetMultiOmicsGDCTrainTest

dataset = DatasetMultiOmicsGDCTrainTest(folder='dataset', omics=('mRNA', 'miRNA',))
dataset.standardize(all_omics=True)
dataset_train, dataset_test = dataset.train_val_test_split()
```

### Training multiple omics at the same time
The `set_omic` method returns a new instance of `DatasetMultiOmics` containing only the selected omic.
Therefore it is possible to have multiple indipendent dataset objects and train multiple omics at the same time

```
from datasets import DatasetMultiOmics

dataset = DatasetMultiOmics(folder='dataset_5000samples', omics=('mRNA', 'meth',), labels='clusters')

mRNA, meth = dataset.set_omic('mRNA'), dataset.set_omic('meth')
# Train on mRNA and meth, e.g. model = train(mRNA, meth)
```

## How to use the Trainer class
This class implements the training procedure for several types of Autoencoders. The requirement is that the model
class specifies which kind of AE it represents, in the form of a static variable called 'ae_type', which has to be one of the following: 'VAE', 'AAE' (Adversarial AE), 'SAAE' (Supervised AAE)
- Initialization:
  The class of course needs a model, optimizer, scheduler and datalaoder.
  There are 2 ways to provide to the class such arguments.
  1. Instantiate them separately and pass them to the constructor:\
     from training import Trainer\
     trainer = Trainer(parameter_dict, model, optimizer, scheduler, train_loader, test_loader)\
     
  2. The class is also able to generate all of those objects on its own, calling 2 methods; one to generate the model and 
     the optimizer, which needs as parameter only the model class; and then one to generate the dataloader, which needs 
     as parameters the dataset class and its arguments. Below is an example:\
     
     from training import Trainer\
     from models import AAE\
     from datasets import DatasetMultiOmics\
     trainer = Trainer(parameter_dict)\
     trainer.generate_model_optimizer(AAE)\
     trainer.generate_dataset_laoder(DatasetMultiOmics, folder='../dataset_folder', omics=['mRNA'])\
     
  In this way the default optimizer used is SGD and the scheduler a StepLR. More choice on that will come in future  
  releases. Note that in both cases a parameter_dict is required to the constructor; an example of that is provided 
  below:
  
  parameters = {\
   'BATCH_SIZE': 64,\
   'LR': 3e-3,\
   'MOMENTUM': 0.9,\
   'WEIGHT_DECAY': 5e-5,\
   'NUM_EPOCHS': 30,\
   'STEP_SIZE': 15,\
   'GAMMA': 0.5,\
   'LOG_FREQUENCY' : 5, # log loss every n epochs\
   'HIDDEN_SIZE':2,\
   'INPUT_SIZE':131 # n. features of the dataset\
  }
  
- Training: 
  To start the training it is sufficient to call the method .train_model(). Loss will be logged as specified in the   
  parameters dict. In this basic version of the class validation is performed only on the VAE. As for the loss functions, 
  they are automatically generated.
 
### Note on training of AAEs and SAAEs

These kind of AEs at training time need to sample from a distribution to which the latent space is forced to coincide.
For this reason the .train_model() method requires as parameter a **sampler** class that exposes a .sample(shape) method.
To conveniently provide this behavior in the package statistic the DistributionSampler class is impelemented.
The DistributionSampler class allows to either create a GaussianMixture distribution and to sample from it, or to provide the model of an AE to be used to sample from the empirical distribution of its latent space.
For now we describe here the first use case.

1. Create a GaussianMixture(GM):
  The class has built-in for convenience 2 2-dimensional GMs, one flower-shaped and one made of uncorrelated 'circles'.
  To list the available distributions:
  from statistic import DistributionSampler\
  sampler = DistributionSampler()\
  sampler.get_default_names()\
  To select one of the defaults (e.g. circles_2d) :\ sampler.build_default('circles_2d')\
  Instead if one wants to build his own customized GM, he only needs to provide the means and covariances tensor.
  If the desired distribution has uncorrelated variables, then it is sufficient to provide the variances.
  
  e.g.This constructs GM in 2D consisting of 5 equally weighted bivariate normal distributions with random means and
  variances\
  sampler.build_gaussian_mixture(torch.randn(5,2), torch.rand(5,2))

2. Inspect the distribution:
  If the distribution created is 2D, the method .plot_sampling(n) will sample n points and plot them.
  There is an additional 'points' parameter: the points whose coordinates it contains will be plotted highlighted in red

3. Sample:
  This is the key feature of the class which is used by the Trainer : the .sample(shape, get_labels=False) method
  allows to sample from the created distribution.
  To the user of the Trainer class that wants to train an AAE/SAAE the use case is the following:
  
  sampler = DistributionSampler()\
  use one of the defaults or build a distribution
  
  trainer = Trainer()\
  initialize the Trainer in the preferred way
  
  trainer.train_model(sampler)
  

## Thanos Wrapper and other trainer classes

The ThanosTrainer classes provides a flexible interface to train any kind of models, by giving the possibility to set strategies composed by an
arbitrary number of steps, each with its own specific Trainer class.
For a multistep training procedure you have 2 ways of providing a Trainer class to take care of it:
- a single class that implements different training stages; and the Thanos wrapper will call them in order by calling the .train_model method with an increasing parameter 'stage=x'
- a different class for each stage; in this case Thanos will call the .train_model on the first one, collect a 'result' which can be any number and type of objects and will pass the result to the second class

The 2 approaches can be combined and have many 'agents' class with many 'stages' each.

### Writing a Trainer classes compliant with the Thanos interface

There are 2 classes available for a new trainer to inherit from, that can provide model and dataset creation, so that the only thing to be added is the
```train_model``` method.

- Inherit from ```biopy.training.Trainer``` -> inheriting from Trainer is a good choice if you have a single model and not, say, one for each omic.
You will automatically have access to data loaders (```self.train_loader, self.test_loader```), your model (```self.model```), and analogously to a scheduler and       optimizer, which will be Adam or SGD based on the parameters set by the user. 

- Inherit from ```biopy.training.JointTrainer``` -> inheriting from JointTrainer is a good choice if you have a set of objects (data loaders, models, optimizers...) for each omic. By subclassing JointTrainer in your ```train_model``` method you will have access to a list of the selected omics (```self.omics```) and to the following dictionaries : 
```
self.train_loaders # and test_loaders as well
self.models
self.optimizers
self.schedulers
```
Each one of these has the omic names as keys for the corresponding object

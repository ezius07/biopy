************
Datasets
************

This section explains how to use the dataset classes.
The `DatasetMultiOmics` class can be used to read the synthetic dataset, while `DatasetMultiOmicsGDC` and `DatasetMultiOmicsGDCTrainTest` can be used to read the GDC dataset downloaded with the aforementioned tool. These classes expect files to be in tsv format with .txt extension and samples along the columns.

Basic synthetic dataset usage
=============================

Toy dataset made up of synthetic data.
In the basic usage scenario, you load all omics and then use one at a time, for instance:

.. code-block:: py3

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


Basic GDC dataset usage
=============================

In order to use the downloaded and processed GDC dataset, `DatasetMultiOmicsGDC` and `DatasetMultiOmicsGDCUnbalanced` classes can be used. These two classes differ from `DatasetMultiOmics` because the label can be programmatically chosen since the downloaded label file is richer amd contains many columns. E.g. For a given experiment, one may want to use the `sample_type` column as a label, but for another experiment the label can be derived from the `project_id`; multiple columns can also be used.

Furthermore:
 - `DatasetMultiOmicsGDC` has to be used when all samples have all the omics and all samples share the same label file, as it happens for the synthetic dataset;
 - `DatasetMultiOmicsGDCUnbalanced` has to be used when instead not all samples have all omics (for each omic there is a separate label file).

E.g. for using the `DatasetMultiOmicsGDCUnbalanced` class with the downloaded GDC dataset, which is "unbalanced" as intended above:

.. code-block:: py3

    from datasets import DatasetMultiOmicsGDCUnbalanced

    # The label here is sample type (2 possible values) + project id (2 possible values), so 4 possible values
    dataset = DatasetMultiOmicsGDCUnbalanced(folder='dataset', omics=('mRNA', 'miRNA',), labels_columns=('sample_type', 'project_id'))

    dataset.standardize(all_omics=True)

    dataset.set_omic('mRNA')
    model1 = train(dataset)

    dataset.set_omic('miRNA')
    model2 = train(dataset)


One shortcoming of the `DatasetMultiOmicsGDCUnbalanced` class is that the `train_val_test_split` method is not available.
However the download script provides a command to split the downloaded dataset in such a way that the test set is made of samples that have all the three omics.
To load it:

.. code-block:: py3

    from datasets import DatasetMultiOmicsGDCUnbalanced

    dataset_train = DatasetMultiOmicsGDCUnbalanced(folder='dataset', omics=('train_mRNA', 'train_miRNA',))
    dataset_test = DatasetMultiOmicsGDC(folder='dataset', omics=('test_mRNA', 'test_miRNA',))

    mean, std = dataset_train.standardize(all_omics=True)
    dataset_test.standardize(all_omics=True, mean=mean, std=std)


The code above is equivalent to the simplified code below which uses the class `DatasetMultiOmicsGDCTrainTest`:

.. code-block:: py3

    from datasets import DatasetMultiOmicsGDCTrainTest

    dataset = DatasetMultiOmicsGDCTrainTest(folder='dataset', omics=('mRNA', 'miRNA',))
    dataset.standardize(all_omics=True)
    dataset_train, dataset_test = dataset.train_val_test_split()
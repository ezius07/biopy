************
Installation
************

One way to reproduce the results presented in this work is to download and run the source code of this project.
We also provide some code and snippets to download and preprocess the presented datasets from their 
original sources so that each dataset can be easiliy and reliably retrieved.


Downloading the source code
===========================

To download the source code, you can run this command:

.. code:: console

    $ git clone https://github.com/BioPyTeam/biopy


.. note::

    Some additional third-party packages must also be installed:
    pytorch, scikit-learn, imblearn, pandas, numpy

	
Downloading the datasets
========================

As also stated elsewhere, the presented methods have been applied on three distinct datasets, two of which
have been already presented and preprocessed in :cite:p:`assignment_paper`.

A549 Dataset
------------

A549 Dataset is a paired multiomics (ATAC-seq and RNA-seq) single-cells dataset comprising A549 cells
of tumoral lung tissue explanted from a 58-year-old caucasian male.
One notable characteristic of this dataset is the fact that it is a paired dataset, which means that
for a given sequencing in a given omic, there is also the corresponding sequencing in the other omic
of the same cell.

ATAC-seq and RNA-seq have been preprocessed also in works other than :cite:p:`assignment_paper`.
So in order to download the preprocessed data from the all different original sources, we suggest
to run this script:

.. code:: console

    $ python3 biopy/utils/download_dataset_nature.py --dataset_dir="dataset_a549"


CD4+ Dataset
------------

This dataset contains two very different omics:
 * Preprocessed RNA single-cell sequencing of naive CD4+ T cells, which have been clustered into two groups: quiescent and poised cells
 * Grayscale 64x64 chromatin images of poised and quiescent single cells
 
This is the main dataset presented in :cite:p:`assignment_paper` and it has been published by the authors on Dropbox.

.. code:: console

    $ wget --content-disposition https://www.dropbox.com/sh/hjt57go4dyahgq7/AAAhAE8bHNn5Sq-D0jGkO_gAa?dl=1
    $ unzip MultiDomainTranslationNatureComm2020.zip
	
GDC Dataset
-----------

We also applied the proposed methods on a preprocessed dataset retrieved from
`The NCI's Genomic Data Commons (GDC) <https://gdc.cancer.gov/>`_

The multiomics dataset contains three omics (mRNA, miRNA and methilation) obtained from 
multicell sequencing of breast tissue.

If you want to recreate the dataset from the GDC portal you can simply run from bash the following command
which will download the data leveraging the GDC API:

.. code:: console

    $ ./download_dataset.sh

If you want to run the provided script step by step, for each action (:code:`download_omic`, ...), there are additional options for customizing directories' locations and other relevant parameters.
See details with :code:`python3 biopy/utils/download_dataset_gdc.py {action} -h` or :code:`python3 biopy/utils/download_dataset_gdc.py -h` to get the list of available actions.

.. warning::
    Even though the final preprocessed and splitted dataset weighs only a few gigabyte,
    the overall data that needs to be downloaded is around 100GB, and at least 350GB after decompression.
    Furthermore, during file downloads, network connections may get terminated, and so the provided
    bash script may error out. However, it can be safely run again after every failure until all files
    have been downloaded. In some cases, additional instruction may be presented to the user on screen
	

	


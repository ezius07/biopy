#!/usr/bin/env bash

echo "Downloading miRNA data..."
python3 biopy/utils/download_dataset_gdc.py download_omic --omic mirna

echo "Downloading mRNA data..."
python3 biopy/utils/download_dataset_gdc.py download_omic --omic mrna

echo "Downloading meth27 data..."
python3 biopy/utils/download_dataset_gdc.py download_omic --omic meth27

echo "Downloading meth450 data..."
python3 biopy/utils/download_dataset_gdc.py download_omic --omic meth450

echo "Preprocessing meth data..."
python3 biopy/utils/download_dataset_gdc.py preprocess_meth

echo "Splitting the dataset in train and test set..."
python3 biopy/utils/download_dataset_gdc.py split_dataset_omics_breast

echo "Cleaning some raw data..."
python3 biopy/utils/download_dataset_gdc.py clean
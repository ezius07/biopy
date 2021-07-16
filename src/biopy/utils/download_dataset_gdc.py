import argparse

import pandas as pd
import numpy as np

import json
import os
import shutil
import requests
import sys
import re
import warnings
import gzip
import hashlib
import openslide
import glob
import math
import itertools

from sklearn.model_selection import StratifiedShuffleSplit
import torch
from functools import reduce

from sklearn.model_selection import train_test_split

filters = {
    "mirna": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content":{
                    "field": "cases.primary_site",
                    "value": ["breast"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_category",
                    "value": ["transcriptome profiling"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_type",
                    "value": ["miRNA Expression Quantification"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                }
            },
        ]
    },
    "mrna": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content":{
                    "field": "cases.primary_site",
                    "value": ["breast"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_category",
                    "value": ["transcriptome profiling"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.analysis.workflow_type",
                    "value": ["HTSeq - FPKM-UQ"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                }
            },
        ]
    },
    "meth27": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content":{
                    "field": "cases.primary_site",
                    "value": ["breast"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_category",
                    "value": ["dna methylation"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.platform",
                    "value": ["illumina human methylation 27"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                }
            },
        ]
    },
    "meth450": {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content":{
                    "field": "cases.primary_site",
                    "value": ["breast"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.data_category",
                    "value": ["dna methylation"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "files.platform",
                    "value": ["illumina human methylation 450"]
                }
            },
            {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                }
            },
        ]
    }
}

def get_file_list(omic):
    url = 'https://api.gdc.cancer.gov/files'
    
    query={
        "filters": filters[omic],
        "format": "json",
        "fields": "file_id,file_name,cases.case_id,cases.samples.sample_type,cases.samples.sample_id,cases.samples.tissue_type,cases.project.project_id,file_size",
        "size": 1000000
    }
    
    result = requests.post(url, json=query)

    result_json = json.loads(result.content)
    
    def map_samples(x):
        if len(x['cases']) != 1:
            print("More than one case for the current file", x)
        assert len(x['cases']) == 1
        
        if len(x['cases'][0]['samples']) > 1:
            if len(set(map(lambda sample: sample['tissue_type'], x['cases'][0]['samples']))) > 1:
                warnings.warn('More than one tissue type for current file' + str(x))
            if len(set(map(lambda sample: sample['sample_type'], x['cases'][0]['samples']))) > 1:
                warnings.warn('More than one sample type for current file' + str(x))

        element = {}
        element['file_id'] = x['file_id']
        element['file_name'] = x['file_name']
        element['case_id'] = x['cases'][0]['case_id']
        element['tissue_type'] = x['cases'][0]['samples'][0]['tissue_type']
        element['sample_type'] = x['cases'][0]['samples'][0]['sample_type']
        element['project_id'] = x['cases'][0]['project']['project_id']
        element['file_size'] = x['file_size']

        element['sample_type'] = 'tumor' if 'tumor' in element['sample_type'].lower() else 'normal'

        return element
    
    files = list(map(map_samples, result_json['data']['hits']))
    files_df = pd.DataFrame(files)
    return files_df

def download_tar(file_ids, download_path):
    url = 'https://api.gdc.cancer.gov/data/'
    query = {"ids": list(file_ids)}
    
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    
    # https://stackoverflow.com/a/1094933
    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)
    
    print("A .tar.gz file is being prepared by the server", flush=True)
    with requests.post(url, json=query, stream=True) as r:
        r.raise_for_status()
        with open(download_path + '.part', 'wb') as f:
            tot_bytes_downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                tot_bytes_downloaded += len(chunk)
                print("Downloaded " + sizeof_fmt(tot_bytes_downloaded) + " " * 20 +"\r", end="", flush=True)
    os.rename(download_path + '.part', download_path)
    print("Downloaded in ", download_path, flush=True)

def extract_tar(download_path):
    print("Extracting", download_path, flush=True)
    if os.system('tar -xf {} --directory {}'.format(download_path, os.path.dirname(download_path))) != 0:
        raise Exception('Failed to extract tar file. Please remove this file and start again: ' + download_path)

def move_downloaded_files(files, omic, download_dir, dataset_dir):
    print("Moving files to dataset dir", flush=True)
    for _, (file_id, file_name, case_id) in files[['file_id', 'file_name', 'case_id']].iterrows():
        src = os.path.join(download_dir, file_id, file_name)
        dst = os.path.join(dataset_dir, case_id, omic, file_name)

        _, file_extension = os.path.splitext(src)

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if file_extension == '.gz':
            with gzip.open(src, 'rb') as f_in:
                dst = dst[:-3]
                with open(dst, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(src)
        else:
            shutil.move(src, dst)

def aggregate_miRNA(files, dataset_dir, omic='mirna'):
    print('Aggregating samples in one file')

    files = files.copy()
    files['aggregated'] = False

    all_samples = None
    for idx, (file_id, file_name, case_id, sample_type) in files[['file_id', 'file_name', 'case_id', 'sample_type']].iterrows():
        example_id = '{}_{}'.format(case_id, sample_type.replace(' ', '-'))
        
        tsv = pd.read_csv(os.path.join(dataset_dir, case_id, omic, file_name), sep='\t')
        tsv = tsv[['miRNA_ID', 'read_count']] \
                .set_index('miRNA_ID') \
                .rename(columns={'read_count': example_id})
        
        if all_samples is None:
            all_samples = tsv
            files.loc[idx, 'aggregated'] = True
        else:
            if not example_id in all_samples:
                all_samples = all_samples.join(tsv, how='outer')
                files.loc[idx, 'aggregated'] = True
            else:
                print("Skipped another example of case " + example_id)
    
    all_samples.to_csv(os.path.join(dataset_dir, 'miRNA.txt'), sep='\t')
    
    files['id'] = files['case_id'] + '_' + files['sample_type'].str.replace(' ', '-')
    files[files['aggregated']][['id', 'file_id', 'case_id', 'sample_type', 'project_id']].to_csv(os.path.join(dataset_dir, 'miRNA_labels.txt'), sep='\t', index=False)

def aggregate_mRNA(files, dataset_dir, omic='mrna'):
    print('Aggregating samples in one file')

    files = files.copy()
    files['aggregated'] = False
    
    all_samples = None
    
    for idx, (file_id, file_name, case_id, sample_type) in files[['file_id', 'file_name', 'case_id', 'sample_type']].iterrows():
        example_id = '{}_{}'.format(case_id, sample_type.replace(' ', '-'))
        
        tsv = pd.read_csv(os.path.join(dataset_dir, case_id, omic, file_name[:-3]), sep='\t', header=None)
        tsv = tsv.rename(columns={0: 'feature', 1: example_id}).set_index('feature') 
        
        if all_samples is None:
            all_samples = tsv
            files.loc[idx, 'aggregated'] = True
        else:
            if not example_id in all_samples:
                all_samples = all_samples.join(tsv, how='outer')
                files.loc[idx, 'aggregated'] = True
            else:
                print("Skipped another example of case " + example_id)
    
    all_samples.to_csv(os.path.join(dataset_dir, 'mRNA.txt'), sep='\t')

    files['id'] = files['case_id'] + '_' + files['sample_type'].str.replace(' ', '-')
    files[files['aggregated']][['id', 'file_id', 'case_id', 'sample_type', 'project_id']].to_csv(os.path.join(dataset_dir, 'mRNA_labels.txt'), sep='\t', index=False)

def aggregate_meth27(files, dataset_dir, omic='meth27'):
    print('Aggregating samples in one file')

    files = files.copy()
    files['aggregated'] = False
    
    all_samples = None
    
    for idx, (file_id, file_name, case_id, sample_type) in files[['file_id', 'file_name', 'case_id', 'sample_type']].iterrows():
        example_id = '{}_{}'.format(case_id, sample_type.replace(' ', '-'))

        tsv = pd.read_csv(os.path.join(dataset_dir, case_id, omic, file_name), sep='\t') \
                .rename(columns={'Beta_value': example_id, 'Composite Element REF': 'Feature'}) \
                .set_index('Feature')
        tsv = tsv[[example_id]]
        
        if all_samples is None:
            all_samples = tsv
            files.loc[idx, 'aggregated'] = True
        else:
            if not example_id in all_samples:
                all_samples = all_samples.join(tsv, how='outer')
                files.loc[idx, 'aggregated'] = True
            else:
                print("Skipped another example of case " + example_id)
    
    all_samples.to_csv(os.path.join(dataset_dir, 'meth27.txt'), sep='\t')

    files['id'] = files['case_id'] + '_' + files['sample_type'].str.replace(' ', '-')
    files[files['aggregated']][['id', 'file_id', 'case_id', 'sample_type', 'project_id']].to_csv(os.path.join(dataset_dir, 'meth27_labels.txt'), sep='\t', index=False)

def aggregate_meth450(files, dataset_dir, omic='meth450'):
    print('Aggregating samples in one file')
    files = files.copy()

    files['aggregated'] = False
    try:
        all_samples = pd.read_csv(os.path.join(dataset_dir, 'meth27.txt'), sep='\t').set_index('Feature')
    except Exception:
        raise Exception('You must download meth27 first!')
        
    for idx, (file_id, file_name, case_id, sample_type) in files[['file_id', 'file_name', 'case_id', 'sample_type']].iterrows():
        example_id = '{}_{}'.format(case_id, sample_type.replace(' ', '-'))

        tsv = pd.read_csv(os.path.join(dataset_dir, case_id, omic, file_name), sep='\t') \
                .rename(columns={'Beta_value': example_id, 'Composite Element REF': 'Feature'}) \
                .set_index('Feature')
        tsv = tsv[[example_id]]
        
        if not example_id in all_samples:
            all_samples = all_samples.join(tsv, how='left')
            files.loc[idx, 'aggregated'] = True
        else:
            print("Skipped another example of case " + example_id)
    
    all_samples.to_csv(os.path.join(dataset_dir, 'meth27-450.txt'), sep='\t')

    files['id'] = files['case_id'] + '_' + files['sample_type'].str.replace(' ', '-')
    files[files['aggregated']][['id', 'file_id', 'case_id', 'sample_type', 'project_id']].to_csv(os.path.join(dataset_dir, 'meth27-450_labels.txt'), sep='\t', index=False)

    # Currently `meth27-450_labels.txt` contains only labels of samples from meth450
    # but meth27-450.txt contains sample from both meth27 and meth450
    # So we update the label file to contain both
    labels27 = pd.read_csv(os.path.join(dataset_dir, 'meth27_labels.txt'), sep='\t')
    labels450 = pd.read_csv(os.path.join(dataset_dir, 'meth27-450_labels.txt'), sep='\t')
    labels27.append(labels450) \
        .drop_duplicates(subset='id', keep='first', inplace=False) \
        .to_csv(os.path.join(dataset_dir, 'meth27-450_labels.txt'), sep='\t', index=False)
 
def download_omic(omic, download_dir, dataset_dir, files_per_tar):
    
    files = get_file_list(omic)

    for file_ids in np.array_split(files['file_id'], len(files['file_id']) // files_per_tar + 1):
        tar_id = hashlib.md5(file_ids.sum().encode('utf-8')).hexdigest()
        download_path = os.path.join(download_dir, tar_id + '.tar.gz')
        if not os.path.isfile(download_path):
            download_tar(file_ids, download_path)
        extract_tar(download_path)

    move_downloaded_files(files, omic, download_dir, dataset_dir)
    
    if omic == 'mirna':
        aggregate_miRNA(files, dataset_dir)
    if omic == 'mrna':
        aggregate_mRNA(files, dataset_dir)
    if omic == 'meth27':
        aggregate_meth27(files, dataset_dir)
    if omic == 'meth450':
        aggregate_meth450(files, dataset_dir)
    
    assert dataset_dir != download_dir, "dataset directory must be different from the download directory"
    shutil.rmtree(download_dir)

    # Cleaning
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def preprocess_meth(in_path='./dataset/meth27-450.txt', out_path='./dataset/meth27-450-preprocessed.txt'):
    meth = pd.read_csv('dataset/meth27-450.txt', sep='\t').set_index('Feature')
    meth_filtered = meth[(meth.isna().sum(axis=1) < 120)]
    meth_filtered_averaged = meth_filtered.T.fillna(meth_filtered.T.mean(axis=0), axis=0).T
    meth_filtered_averaged.to_csv(out_path, sep='\t')
    os.rename(os.path.join(os.path.dirname(in_path),os.path.basename(in_path).split(".")[-2]+"_labels.txt"),os.path.join(os.path.dirname(out_path),os.path.basename(out_path).split(".")[-2]+"_labels.txt"))

def split_dataset_omics_breast(dataset_dir='./dataset/', test_size=0.2, output_dataset_dir='./dataset/', omics=('miRNA', 'mRNA', 'meth27-450-preprocessed')):
    """
        Balanced train/test split.
        In the validation set, only a subset of samples that have all omics are considered
        (Since there are very few normal tissue samples with all omics, all those normal samples are considered;
        The proportion of 1:10 of normal/tumoral that is present in the training set is preserved anyways)
    """

    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # Join all labels of all omics
    labels = []
    for omic in omics:
        labels.append(pd.read_csv(os.path.join(dataset_dir, omic + '_labels.txt'), sep='\t').set_index('id'))

    for i, label in enumerate(labels[1:]):
        labels[0] = labels[0].join(label, how='inner', rsuffix=str(i))

    labels = labels[0]
    labels['label'] = labels['project_id'] + '_' + labels['sample_type']

    # We take some random samples in a stratified manner
    _, test_samples = train_test_split(
        labels.index,
        test_size=test_size,
        stratify=labels['label'], shuffle=True, random_state=42)
    
    print('Label distribution in the test set')
    print(labels['label'][test_samples].value_counts())

    for omic in omics:
        omic_df = pd.read_csv(os.path.join(dataset_dir, omic + '.txt'), sep='\t')
        omic_labels_df = pd.read_csv(os.path.join(dataset_dir, omic + '_labels.txt'), sep='\t').set_index('id')

        omic_train_samples = omic_df.columns[~omic_df.columns.isin(test_samples)] # It already includes the Feature column
        omic_test_samples = omic_df.columns[[0]].append(test_samples)

        train_df = omic_df[omic_train_samples]
        test_df = omic_df[omic_test_samples]

        # Discard features (rows) with one unique value
        relevant_features = train_df.groupby(train_df.columns[0]).filter(lambda x: len(set(x.iloc[0].values.tolist()[1:])) > 1).index

        train_df = train_df.loc[relevant_features]
        test_df = test_df.loc[relevant_features]

        os.remove(os.path.join(dataset_dir, omic + '.txt'))
        os.remove(os.path.join(dataset_dir, omic + '_labels.txt'))
        train_df.to_csv(os.path.join(output_dataset_dir, 'train_' + omic + '.txt'), sep='\t', index=False)
        omic_labels_df.loc[omic_train_samples[1:]].to_csv(os.path.join(output_dataset_dir, 'train_' + omic + '_labels.txt'), sep='\t', index=True)
        test_df.to_csv(os.path.join(output_dataset_dir, 'test_' + omic + '.txt'), sep='\t', index=False)

    labels.loc[test_samples][['file_id', 'case_id', 'sample_type', 'project_id']].to_csv(os.path.join(output_dataset_dir, 'test_labels.txt'), sep='\t', index=True, index_label='id')


def clean(root):
    for filename in os.listdir(root):
        file_path = os.path.join(root, filename)
        if "train" not in filename and "test" not in filename:
            os.remove(file_path)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='action', required=True)

    download_omic_parser = subparsers.add_parser('download_omic')
    download_omic_parser.add_argument('--omic', type=str, required=True, choices = ["mirna", "mrna", "meth27", "meth450"],
                                      help='Omic of the data')
    download_omic_parser.add_argument('--download_dir', type=str, default='./downloaded_files/',
                                      help='Directory where temporary download data is stored')
    download_omic_parser.add_argument('--dataset_dir', type=str, default='./dataset/',
                                      help='Directory where the final dataset is stored')
    download_omic_parser.add_argument('--files_per_tar', type=int, default=50,
                                      help='Approx. number of sample for each tar that will be downloaded')
    
    preprocess_meth_parser = subparsers.add_parser('preprocess_meth')
    preprocess_meth_parser.add_argument('--in_path', type=str, default='./dataset/meth27-450.txt')
    preprocess_meth_parser.add_argument('--out_path', type=str, default='./dataset/meth27-450-preprocessed.txt')

    split_omics_parser = subparsers.add_parser('split_dataset_omics_breast')
    split_omics_parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    split_omics_parser.add_argument('--output_dataset_dir', type=str, default='./dataset/')
    split_omics_parser.add_argument('--test_size', type=float, default=0.2)
    split_omics_parser.add_argument('--omics', nargs='+', default=('miRNA', 'mRNA', 'meth27-450-preprocessed'), help="List of omics splitted")

    clean_parser = subparsers.add_parser('clean')
    clean_parser.add_argument('--root', type=str, default='./dataset/')

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('action')](**kwargs)
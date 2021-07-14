# -*- coding: utf-8 -*-

RAW_DATASET = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/GSE117089/suppl/GSE117089_RAW.tar", "GSE117089_RAW.tar", "Raw dataset containing unprocessed ATAC and RNA samples")
RNA_Q_VALUES = ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6571013/bin/NIHMS1033869-supplement-Supplimentary_table.xlsx", "NIHMS1033869-supplement-Supplimentary_table.xlsx", "Precomputed q-values for RNA genes")
ATAC_MOTIF = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20249-2/MediaObjects/41467_2020_20249_MOESM4_ESM.xlsx", "41467_2020_20249_MOESM4_ESM.xlsx", "Preprocessed ATAC motif")
FILES = (RAW_DATASET, RNA_Q_VALUES, ATAC_MOTIF)

import requests
import os
from scipy.io import mmread
import pandas as pd
import gzip
import argparse

def download_file(url, download_path, description):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    
    # https://stackoverflow.com/a/1094933
    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)
    
    print("Downloading:", description, flush=True)
    with requests.get(url, stream=True) as r:
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

def main(download_dir, dataset_dir):
    assert dataset_dir != download_dir, "dataset directory must be different from the download directory"
    
    print("Downloading files")
    for file in FILES:
        url, filename, description = file
        download_path = os.path.join(download_dir, filename)
        if not os.path.isfile(download_path):
            download_file(url, download_path, description)

    # Extract raw dataset
    extract_tar(os.path.join(download_dir, RAW_DATASET[1]))

    print("Loading ATAC data")
    atac = pd.read_excel(os.path.join(download_dir, "41467_2020_20249_MOESM4_ESM.xlsx"), sheet_name=0, header=0)
    atac = atac.set_index(atac.columns[0])
    atac_sample_ids = atac.columns

    print("Loading RNA data:")

    print(" - Differential genes expressions for evaluating relevant genes for RNA omic")
    differential_genes_expr = pd.read_excel(os.path.join(download_dir, "NIHMS1033869-supplement-Supplimentary_table.xlsx"), sheet_name="Table S2", header=1)
    relevant_genes = differential_genes_expr[differential_genes_expr["qval"] < .05]["gene_id"]

    print(" - RNA expr value matrix")
    matrix = mmread(gzip.open(os.path.join(download_dir, "GSM3271040_RNA_sciCAR_A549_gene_count.txt.gz"))).tocsr()

    print(" - RNA value matrix metadata (row and cols names)")
    cells = pd.read_csv(gzip.open(os.path.join(download_dir, "GSM3271040_RNA_sciCAR_A549_cell.txt.gz")))
    genes = pd.read_csv(gzip.open(os.path.join(download_dir, "GSM3271040_RNA_sciCAR_A549_gene.txt.gz")))
    cells = cells.reset_index().rename(columns={'index': 'idx'}).set_index('sample')
    genes = genes.reset_index().rename(columns={'index': 'idx'}).set_index('gene_id')

    print("Saving dataset")

    os.makedirs(dataset_dir, exist_ok=True)

    # RNA
    matrix_rows = genes.loc[relevant_genes]["idx"]
    matrix_rows_names = genes.loc[relevant_genes]["gene_short_name"]
    matrix_cols = cells.loc[atac_sample_ids]["idx"]
    matrix_cols_names = cells.loc[atac_sample_ids].index
    df = pd.DataFrame(matrix[matrix_rows, :][:, matrix_cols].todense(), columns=matrix_cols_names, index=matrix_rows_names)
    df.to_csv(os.path.join(dataset_dir, 'rna.txt'), sep='\t', index=True)

    # ATAC
    atac.index.name = 'gene_short_name'
    atac.to_csv(os.path.join(dataset_dir, 'atac.txt'), sep='\t', index=True)

    # LABELS
    cells.loc[atac_sample_ids][["treatment_time"]].to_csv(os.path.join(dataset_dir, 'labels.txt'), sep='\t', index=True, index_label='id')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_dir', type=str, default='./downloaded_files/', help='Directory where temporary download data is stored')
    parser.add_argument('--dataset_dir', type=str, default='./dataset_nature_atac-rna/', help='Directory where the final dataset is stored')

    main(**vars(parser.parse_args()))
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import os
import glob
import torch

from sklearn.manifold import TSNE

from biopy.datasets import DatasetMultiOmicsGDCTrainTest, DatasetMultiOmicsNatureA549, DatasetMultiOmicsNatureTrainTest
from biopy.utils import sample_encodings
from biopy.models import FC_VAE, NucleiImgVAE

def get_dataset(dataset_name, dataset_folder):
    
    def gdc():
        ds = DatasetMultiOmicsGDCTrainTest(folder=dataset_folder)
        ds.set_omic('mRNA')
        ds.log1p(all_omics=False)
        ds.standardize(all_omics=True)
        ds.feature_selection_by('mad', all_omics=True, top_n=2000)
        
        legend_labels = ['', '', '', '', 'normal', 'tumor']
        palette = ['#0d47a1', '#b71c1c', '#2196f3', '#f44336', '#90caf9', '#ef9a9a']
        omics = ['mRNA', 'miRNA', 'meth27-450-preprocessed']
        
        return ds, omics, (legend_labels, palette)
    
    def cd4():
        ds = DatasetMultiOmicsNatureTrainTest(folder=dataset_folder)
        
        legend_labels = ['', '', 'poised', 'quiesc.']
        palette = ['#E98616', '#8616E9', '#A86214', '#4d0f85']
        omics = ['nuclei-images', 'rna']
        
        return ds, omics, (legend_labels, palette)
    
    def a549():
        ds = DatasetMultiOmicsNatureA549(folder=dataset_folder)
        ds.log1p(all_omics=True)
        ds.standardize(all_omics=True)
        
        legend_labels = ['', '', '', '0 hours', '1 hours', '3 hours']
        palette = ['#E98616', '#16E986', '#8616E9', '#A86214','#14A862', '#4d0f85']
        omics = ['rna', 'atac']
        
        return ds, omics, (legend_labels, palette)
    
    # Add here your additional datasets and/or preprocessings
    
    return locals()[dataset_name]()

def get_models(ds, omics, model_name):
    
    def gdc_vae():
        return {omic: FC_VAE(data_size=ds.set_omic(omic)[0][0].shape[0],
                             hidden_size=80, n_hidden=ds.set_omic(omic)[0][0].shape[0], 
                             last_hidden=100)
               for omic in omics}
    
    def cd4_vae():
        return {'nuclei-images': NucleiImgVAE(data_size=1, hidden_size=128),
                'rna': FC_VAE(data_size=7633, hidden_size=128, n_hidden=1024),
               }
    
    def a549_vae():
        return {'atac': FC_VAE(data_size=815, n_hidden=815, hidden_size=50, last_hidden=100),
                'rna': FC_VAE(data_size=2613, n_hidden=2613, hidden_size=50, last_hidden=100),
               }
    
    # Add here your additional models
      
    return locals()[model_name]()
    
def get_latent_encodings(model, dataset, tsne=True, dim=2):
    
    assert len(model) == len(dataset)
    encodings = []
    for i, mod in enumerate(model):
        encodings.append(sample_encodings(mod, dataset[i], tsne=False))
    encodings_py = np.concatenate(encodings)
    if tsne:
        encodings_py = TSNE(n_components=dim).fit_transform(encodings_py)

    lab = []
    for i, data in enumerate(dataset):
        if hasattr(data, 'no_slicing'):
            nlab = len(set(list(map(lambda x: x[1].numpy().item(), data))))
            lab.append(np.array(list(map(lambda x: x[1] + (i * nlab), data))))
        else:
            nlab = len(set(data[0][:][1].numpy()))
            lab.append(np.array([int(item[1]) + (i * nlab) for item in data]))
    colors = np.concatenate(lab)

    return encodings_py, colors

def main(dataset_name, dataset_folder, model_name, checkpoints, output_path):
    
    ds, omics, (legend_labels, palette) = get_dataset(dataset_name, dataset_folder)
    models = get_models(ds, omics, model_name)
    datasets = []
    
    for omic in omics:
        model = models[omic]
        model.load_state_dict(torch.load(checkpoints.format(omic=omic), map_location=torch.device('cpu')))
        model.eval()
        d = ds.set_omic(omic)
        d.no_slicing = 'YES'
        datasets.append(d)

    encodings_py, colors = get_latent_encodings(model=[models[omic] for omic in omics], dataset=datasets)

    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(encodings_py[:, 0], encodings_py[:, 1], c=colors, s=4,
                    cmap=mpl.colors.ListedColormap(palette))

    ax.set_axis_off()
    ax.set_xlim([-75, 75])

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Save the legend on a separate file
    fig, ax = plt.subplots(figsize=(1, 1))
    legend = ax.legend(sc.legend_elements()[0], legend_labels, loc='upper left',
                       title="  OMIC     LABEL ", ncol=len(omics), columnspacing=-1)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    ax.set_axis_off()
    ax.set_xlim([-75, 75])
    fig.savefig(f"{output_path}.legend.png", dpi=300, bbox_inches='tight')

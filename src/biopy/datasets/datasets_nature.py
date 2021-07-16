import torch
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage import io

import os

from torch.utils.data import TensorDataset
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    def SMOTE(*args, **kwargs):
        raise Exception('+++ Please install imblearn +++')

class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, sample):
        image_tensor = sample['image_tensor']
 
        # rescale by maximum and minimum of the image tensor
        minX = image_tensor.min()
        maxX = image_tensor.max()
        image_tensor=(image_tensor-minX)/(maxX-minX)
 
        # resize the inputs
        # torch image tensor expected for 3D operations is (N, C, D, H, W)
        image_tensor = image_tensor.max(axis=0)
        image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image_tensor = np.clip(image_tensor, 0, 1)
        return torch.from_numpy(image_tensor).view(1, 64, 64)


class NucleiDatasetNew(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize(), **kwargs):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform
        self.threshold = 0.74

    # Utility function to load images from a HDF5 file
    def load_images(self):
        # load labels
        label_data = pd.read_csv(os.path.join(self.datadir, "ratio.csv"))
        label_data_2 = pd.read_csv(os.path.join(self.datadir, "protein_ratios_full.csv"))
        label_data = label_data.merge(label_data_2, how='inner', on='Label')
        label_dict = {name: (float(ratio), np.abs(int(cl)-2)) for (name, ratio, cl) in zip(list(label_data['Label']), list(label_data['Cor/RPL']), list(label_data['mycl']))}
        label_dict_2 = {name: np.abs(int(cl)-2) for (name, cl) in zip(list(label_data_2['Label']), list(label_data_2['mycl']))}
        del label_data
        del label_data_2

        # load images
        images_train = []
        images_test = []

        for f in os.listdir(os.path.join(self.datadir, "images")):
            basename = os.path.splitext(f)[0]
            fname = os.path.join(os.path.join(self.datadir, "images"), f)
            if basename in label_dict.keys():
                images_test.append({'name': basename, 'label': label_dict[basename][0], 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict[basename][1]})
            else:
                try:
                    images_train.append({'name': basename, 'label': -1, 'image_tensor': np.float32(io.imread(fname)), 'binary_label': label_dict_2[basename]})
                except Exception as e:
                    pass

        if self.mode == 'train':
            return images_train
        elif self.mode == 'test':
            return images_test
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]

        if self.transform:
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {'image_tensor': image_tensor, 'name': sample['name'], 'label': sample['label'], 'binary_label': sample['binary_label']}
        return sample


class ATAC_Dataset(Dataset):
    def __init__(self, datadir, **kwargs):
        self.datadir = datadir
        self.atac_data, self.labels = self._load_atac_data()

    def __len__(self):
        return len(self.atac_data)

    def __getitem__(self, idx):
        atac_sample = self.atac_data[idx]
        cluster = self.labels[idx]
        return {'tensor': torch.from_numpy(atac_sample).float(), 'binary_label': int(cluster)}

    def _load_atac_data(self):
        data = pd.read_csv(os.path.join(self.datadir, "df_peak_counts_names_nCD4_seuratnorm.csv"), index_col=0)
        data = data.transpose()
        labels = pd.read_csv(os.path.join(self.datadir, "clustlabels_peak_counts_names_nCD4_seurat_n_2.csv"), index_col=0)

        data = labels.merge(data, left_index=True, right_index=True)
        data = data.values

        return data[:,1:], data[:,0]


class RNA_Dataset(Dataset):
    def __init__(self, datadir, **kwargs):
        self.datadir = datadir
        self.rna_data, self.labels = self._load_rna_data()

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        cluster = self.labels[idx]
        coro1a = rna_sample[5849]
        rpl10a = rna_sample[2555]
        return {'tensor': torch.from_numpy(rna_sample).float(), 'coro1a': coro1a, 'rpl10a': rpl10a, 'label': coro1a/rpl10a, 'binary_label': int(cluster)}

    def _load_rna_data(self):
        data = pd.read_csv(os.path.join(self.datadir, "filtered_lognuminorm_pc_rp_7633genes_1396cellsnCD4.csv"), index_col=0)
        data = data.transpose()
        labels = pd.read_csv(os.path.join(self.datadir, "labels_nCD4_corrected.csv"), index_col=0)

        data = labels.merge(data, left_index=True, right_index=True)
        data = data.values

        return data[:,1:], np.abs(data[:,0]-1)

class NucleiDatasetWrapper(Dataset):

    def __init__(self, raw_idx=False, *args, **kwargs):
        self.dataset = NucleiDatasetNew(*args, **kwargs)
        self.rebalanced_data = None # Important: must come before any dataset access
        self.raw_idx = raw_idx
        self.labels = torch.tensor([self[x][1] for x in range(len(self))], dtype=torch.int64)
        self.omic_selected = 'nuclei-images'

    def SMOTE(self, **kwargs):
        data = np.array(list(map(lambda x: x[0].flatten().numpy(), self)))
        labels = self.labels.numpy()
        data = data.reshape(data.shape[0], -1) # flatten

        res, res_y = SMOTE(**kwargs).fit_resample(data, labels)
        self.rebalanced_data = torch.tensor(res.reshape(res.shape[0], 1, 64, 64), dtype=torch.float32)
        self.labels = torch.tensor(res_y, dtype=torch.int64)

    def __len__(self):
        return len(self.dataset) if self.rebalanced_data is None else len(self.rebalanced_data)

    def __getitem__(self, idx):
        if self.rebalanced_data is None:
            el = self.dataset[idx]
            if self.raw_idx:
                return el
            return el['image_tensor'], torch.tensor(int(el['binary_label'])).long()
        else:
            return self.rebalanced_data[idx], self.labels[idx]

class OmicDatasetWrapper(Dataset):

    def __init__(self, dataset_class, raw_idx=False, *args, **kwargs):
        self.dataset = dataset_class(*args, **kwargs)
        self.raw_idx = raw_idx
        self.labels = torch.tensor(np.array([self[x][1] for x in range(len(self))], dtype=np.int64), dtype=torch.int64)
        self.omic_selected = 'rna'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        el = self.dataset[idx]
        if self.raw_idx:
            return el
        return el['tensor'], torch.tensor(int(el['binary_label'])).long()

    def train_val_test_split(self, test=.2, validation=0, random_state=97):
        sss_tvt = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=random_state)
        train_val_indices, test_indices = next(sss_tvt.split(np.zeros(len(self.labels)), self.labels))
        subset_train, subset_test = Subset(self, train_val_indices), Subset(self, test_indices)
        
        subset_train.omic_selected = self.omic_selected
        subset_test.omic_selected = self.omic_selected

        subset_train.labels = self.labels[train_val_indices]
        subset_test.labels = self.labels[test_indices]

        subset_train.no_slicing = True
        subset_test.no_slicing = True
        
        return subset_train, subset_test

class DatasetMultiOmicsNatureTrainTest:

    def __init__(self, folder, load_omics_from_disk=True, **kwargs):
        if load_omics_from_disk:
            self.load_omics_from_disk(folder, **kwargs)
        else:
            self.load_omics_from_dict(**kwargs)
        self.labels_ids = {0: 'poised', 1: 'quiescent'}
    
    def load_omics_from_disk(self, folder, **kwargs):
        omics = kwargs.pop('omics', ('nuclei-images', 'rna', 'atac'))
                
        self.datasets = {}
        if 'nuclei-images' in omics:
            self.datasets['train_nuclei-images'] = NucleiDatasetWrapper(datadir=os.path.join(folder, "nuclear_crops_all_experiments"), mode='train', **kwargs)
            self.datasets['test_nuclei-images'] = NucleiDatasetWrapper(datadir=os.path.join(folder, "nuclear_crops_all_experiments"), mode='test', **kwargs)
        if 'rna' in omics:
            self.datasets['rna'] = OmicDatasetWrapper(RNA_Dataset, datadir=os.path.join(folder, "nCD4_gene_exp_matrices"))
        if 'atac' in omics:
            self.datasets['atac'] = OmicDatasetWrapper(ATAC_Dataset, datadir=os.path.join(folder, "atac_seq_data"))

        self.omic_selected = omics[0]

    def load_omics_from_dict(self, datasets, **kwargs):
        self.datasets = datasets
        self.omic_selected = next(iter(datasets.keys()))

    @property
    def omic_selected(self):
        return self._omic_selected

    @omic_selected.setter
    def omic_selected(self, omic):
        self._omic_selected = omic
        if omic != 'nuclei-images':
            self.labels = self.datasets[self._omic_selected].labels
        else:
            nuclei_images_labels = list(map(lambda x: self.datasets[x].labels, [k for k in self.datasets.keys() if 'nuclei-images' in k]))
            self.labels = torch.cat(nuclei_images_labels)

    def set_omic(self, omic):
        self.omic_selected = omic
        if omic == 'nuclei-images':
            datasets = {k: self.datasets[k] for k in self.datasets.keys() if 'nuclei-images' in k}
        else:
            datasets = {omic: self.datasets[omic]}
        return DatasetMultiOmicsNatureTrainTest(folder=None, load_omics_from_disk=False, datasets=datasets)

    def log1p(self, *args, **kwargs):
        pass

    def standardize(self, **kwargs):
        pass

    def minmax_scale(self, **kwargs):
        pass

    def feature_selection_by(self, *args, **kwargs):
        pass

    def SMOTE(self, all_omics, **kwargs):
        if 'train_nuclei-images' in self.datasets:
            self.datasets['train_nuclei-images'].SMOTE(**kwargs)

    def train_val_test_split(self, **kwargs):
        if 'nuclei-images' in self._omic_selected:
            return self.datasets['train_nuclei-images'], self.datasets['test_nuclei-images']
        else:
            return self.datasets[self._omic_selected].train_val_test_split()

    def __len__(self):
        if self._omic_selected != 'nuclei-images':
            return len(self.datasets[self._omic_selected])
        raise Exception('Not implemented for nuclei-images')

    def __getitem__(self, idx):
        if self._omic_selected != 'nuclei-images':
            return self.datasets[self._omic_selected][idx]
        raise Exception('Not implemented for nuclei-images')
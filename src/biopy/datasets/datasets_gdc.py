import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import StratifiedShuffleSplit

from functools import reduce

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    def SMOTE(*args, **kwargs):
        raise Exception('+++ Please install imblearn +++')

class DatasetMultiOmics(Dataset):

    def load_tensor(self, csv_path, dtype=np.float32):
        dataset = pd.read_csv(csv_path, sep='\t')
        dataset = dataset.transpose()
        dataset = dataset.rename(columns=dataset.iloc[0])[1:]
        tensor = torch.tensor(dataset.to_numpy(dtype=dtype))
        return tensor

    def load_omics_from_disk(self, folder='dataset_5000samples', omics=('mRNA', 'meth', 'prot'), labels='clusters', **kwargs):
        """
            Omics names can be prefixed with a string followed by a _ and the prefix will be removed
            E.g. train_mRNA -> internally stored as mRNA.
            Thus omic names cannot contain underscores.
        """

        self.omics = {}
        
        # Take first because shape is [1, 5000]
        self.labels = self.load_tensor(os.path.join(folder, labels + '.txt'), dtype=np.int)[0] if not labels is None else None

        for omic in omics:
            tensor = self.load_tensor(os.path.join(folder, omic + '.txt'))
            omic_name = omic.split('_')[-1] # remove train_ or test_ prefix if present
            self.omics[omic_name] = tensor

        self.omic_selected = next(iter(self.omics.keys()))

    def load_omics_from_dict(self, omics=None, labels=None, **kwargs):
        self.omics = omics
        self.labels = labels

        self.omic_selected = next(iter(omics.keys()))

    def __init__(self, load_omics_from_disk=True, **kwargs):
        if load_omics_from_disk:
            self.load_omics_from_disk(**kwargs)
        else:
            self.load_omics_from_dict(**kwargs)

    @property
    def omic_selected(self):
        return self._omic_selected

    @omic_selected.setter
    def omic_selected(self, omic):
        if omic not in self.omics:
            raise ValueError(f'Wrong omic, {omic} is not present among the loaded omics')
        self._omic_selected = omic
    
    def set_omic(self, omic):
        # Call this method only from outside this class
        self.omic_selected = omic
        return DatasetMultiOmics(load_omics_from_disk=False, omics={omic: self.omics[omic]}, labels=self.labels)
        
    def train_val_test_split(self, test=.2, validation=0, random_state=97):
        sss_tvt = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=random_state)
        if validation > 0:
            sss_tv = StratifiedShuffleSplit(n_splits=1, test_size=validation / (1 - test), random_state=random_state)

        # From the documentation:
        # Providing y is sufficient to generate the splits and hence np.zeros(n_samples) 
        # is be used as a placeholder for X instead of actual training data.
        train_val_indices, test_indices = next(sss_tvt.split(np.zeros(len(self.labels)), self.labels))

        if validation > 0:
            train_indices, val_indices = next(sss_tv.split(np.zeros(len(train_val_indices)), self.labels[train_val_indices]))
            
            return Subset(self, train_indices), Subset(self, val_indices), Subset(self, test_indices)

        #return Subset(self, train_val_indices), Subset(self, test_indices)
        ds_train = DatasetMultiOmics(load_omics_from_disk=False, 
                                     omics={omic: self.omics[omic][train_val_indices] for omic in self.omics.keys()}, 
                                     labels=self.labels[train_val_indices])
        ds_test = DatasetMultiOmics(load_omics_from_disk=False, 
                                     omics={omic: self.omics[omic][test_indices] for omic in self.omics.keys()}, 
                                     labels=self.labels[test_indices])

        return ds_train, ds_test

    def log1p(self, all_omics=False):
        if not all_omics:
            self.omics[self.omic_selected] = self.omics[self.omic_selected].log1p()
        else:
            for i, data in self.omics.items():
                self.omics[i] = self.omics[i].log1p()

    def standardize(self, all_omics=False, mean=None, std=None):
        if not all_omics:
            data = self.omics[self.omic_selected]
            if mean is None:
                mean, std = data.mean(axis=0), data.std(axis=0) 
            res = (data - mean) / std
            self.omics[self.omic_selected] = res
        else:
            if mean is None:
                mean, std = {}, {}
                for i, data in self.omics.items():
                    mean[i] = data.mean(axis=0)
                    std[i] = data.std(axis=0)
            for i, data in self.omics.items():
                res = (data - mean[i]) / std[i]
                self.omics[i] = res

        return mean, std

    def minmax_scale(self, all_omics=False, min=None, max=None):
        if not all_omics:
            data = self.omics[self.omic_selected]
            if min is None:
                min, max = data.min(axis=0).values, data.max(axis=0).values
            res = (data - min) / (max - min)
            self.omics[self.omic_selected] = res
        else:
            if min is None:
                min, max = {}, {}
                for i, data in self.omics.items():
                    min[i] = data.min(axis=0).values
                    max[i] = data.max(axis=0).values
            for i, data in self.omics.items():
                res = (data - min[i]) / (max[i] - min[i])
                self.omics[i] = res

        return min, max

    def feature_selection_by(self, by, all_omics=False, top_n=100, indices=None):
        if by not in {'mad', 'std'}:
            raise ValueError(f'Only by mad or std are supported, given {by}')

        def _get_top_n_features_by_mad(omic_selected):
            data = self.omics[omic_selected]
            medians, _ = data.median(axis=0)
            data_, _ = (data - medians).abs().median(axis=0)
            indices = data_.argsort(axis=0, descending=True)
            indices = indices[:top_n] if top_n < indices.shape[0] else indices
            indices, _ = indices.sort()
            return indices

        def _get_top_n_features_by_std(omic_selected):
            data = self.omics[omic_selected]
            indices = data.std(axis=0).argsort(axis=0, descending=True)
            indices = indices[:top_n] if top_n < indices.shape[0] else indices
            indices, _ = indices.sort()
            return indices
        
        _get_top_n_features = locals()[f'_get_top_n_features_by_{by}']

        if not all_omics:
            if indices is None:
                top_n_feats_idx = _get_top_n_features(self.omic_selected)
                indices = top_n_feats_idx
            self.omics[self.omic_selected] = self.omics[self.omic_selected][:, indices]
        else:
            if indices is None:
                indices = {}
                for omic_selected in self.omics:
                    top_n_feats_idx = _get_top_n_features(omic_selected)
                    indices[omic_selected] = top_n_feats_idx
            for omic_selected in self.omics:
                self.omics[omic_selected] = self.omics[omic_selected][:, indices[omic_selected]]

        return indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        item = self.omics[self._omic_selected][idx] # For performance reasons, use the protected property
  
        return item, label

class DatasetMultiOmicsGDC(DatasetMultiOmics):
    """
        Load omics from preprocessed GDC dataset.
        This class differs from `DatasetMultiOmics` because the label file is richer and differently formatted
    """

    def load_labels(self, csv_path, labels_columns, labels_ids):
        labels_ids = labels_ids.values() if isinstance(labels_ids, dict) else labels_ids

        tsv = pd.read_csv(csv_path, sep='\t')
        labels = reduce(lambda a,b: a + b, [tsv[labels_column] for labels_column in labels_columns])
        labels = pd.Series(pd.Categorical(labels, categories=labels_ids))

        labels_values = torch.tensor(labels.cat.codes, dtype=torch.long)
        labels_ids = dict(zip(range(len(labels.cat.categories)), labels.cat.categories))

        return labels_values, labels_ids

    def load_omics_from_disk(self, folder='dataset', omics=('test_miRNA', 'test_mRNA', 'test_meth27-450-preprocessed'), labels='test_labels', labels_columns=('sample_type', 'project_id'), labels_ids=None, **kwargs):
        super().load_omics_from_disk(folder=folder, omics=omics, labels=None, **kwargs)

        self.labels_ids = labels_ids

        if not labels is None:
            self.labels, self.labels_ids = self.load_labels(os.path.join(folder, labels + '.txt'), labels_columns, self.labels_ids)


class DatasetMultiOmicsGDCUnbalanced(DatasetMultiOmicsGDC):
    """
        Load omics from preprocessed GDC dataset with different samples per omic.
        This is useful if the different omics do not share the same patients.
        
        E.g. miRNA has got 300 samples and mRNA has got 400 samples.

        This is the case of our GDC training split.
    """

    def load_omics_from_disk(self, folder='dataset', omics=('train_miRNA', 'train_mRNA', 'train_meth27-450-preprocessed'), labels_columns=('sample_type', 'project_id'), labels_ids=None, **kwargs):     
        super().load_omics_from_disk(folder=folder, omics=omics, labels=None, **kwargs)

        self.labels_ids = labels_ids

        self._omics_labels = {}

        for omic in omics:
            omic_labels, self.labels_ids = self.load_labels(os.path.join(folder, omic + '_labels.txt'), labels_columns, self.labels_ids)
            self._omics_labels[omic.split('_')[-1]] = omic_labels

        self.omic_selected = next(iter(self.omics.keys()))

    @DatasetMultiOmicsGDC.omic_selected.setter
    def omic_selected(self, omic):
        if not hasattr(self, '_omics_labels'):
            return # Parent class set omic_selected but this class is not ready yet to handle it

        DatasetMultiOmicsGDC.omic_selected.fset(self, omic)
        self.labels = self._omics_labels[omic]

    def SMOTE(self, all_omics=True, **kwargs):
        if not all_omics:
            res, res_y = SMOTE(**kwargs).fit_resample(self.omics[self.omic_selected], self._omics_labels[self.omic_selected])
            self.omics[self.omic_selected] = torch.tensor(res, dtype=torch.float)
            self._omics_labels[self.omic_selected] = torch.tensor(res_y, dtype=torch.long)
            self.omic_selected = self.omic_selected # We must update self.labels
        else:
            for i, data in self.omics.items():
                res, res_y = SMOTE(**kwargs).fit_resample(self.omics[i], self._omics_labels[i])
                self.omics[i] = torch.tensor(res, dtype=torch.float)
                self._omics_labels[i] = torch.tensor(res_y, dtype=torch.long)
            self.omic_selected = self.omic_selected # We must update self.labels

    def train_val_test_split(self, *args, **kwargs):
        raise Exception('Splitting this dataset is not supported')

class DatasetMultiOmicsGDCTrainTest:
    """
        Utility class to load an already splitted GDC dataset
        where the training set is 'unbalanced' and the test set is not.

        The preprocessing operations (e.g standardization) are always fitted
        on the training set only and then applied separately on training and on test set.
    """
    
    def load_omics_from_disk(self, **kwargs):
        omics = kwargs.pop('omics', ('miRNA', 'mRNA', 'meth27-450-preprocessed'))

        self.ds_train = DatasetMultiOmicsGDCUnbalanced(omics=['train_' + omic for omic in omics], **kwargs)

        if 'labels_ids' not in kwargs:
            kwargs['labels_ids'] = self.ds_train.labels_ids
        self.ds_test = DatasetMultiOmicsGDC(omics=['test_' + omic for omic in omics], **kwargs)

        self.labels_ids = self.ds_train.labels_ids

    def load_omics_from_dict(self, ds_train=None, ds_test=None, **kwargs):
        self.ds_train = ds_train
        self.ds_test = ds_test

    def __init__(self, load_omics_from_disk=True, **kwargs):
        if load_omics_from_disk:
            self.load_omics_from_disk(**kwargs)
        else:
            self.load_omics_from_dict(**kwargs)

    @property
    def labels(self):
        return torch.cat([self.ds_train.labels, self.ds_test.labels])

    @property
    def omic_selected(self):
        return self.ds_train.omic_selected

    def set_omic(self, omic):
        ds_train_ = self.ds_train.set_omic(omic)
        ds_test_ = self.ds_test.set_omic(omic)
        return DatasetMultiOmicsGDCTrainTest(load_omics_from_disk=False, ds_train=ds_train_, ds_test=ds_test_)

    def log1p(self, *args, **kwargs):
        self.ds_train.log1p(*args, **kwargs)
        self.ds_test.log1p(*args, **kwargs)

    def standardize(self, **kwargs):
        mean, std = self.ds_train.standardize(**kwargs)
        
        kwargs['mean'] = mean
        kwargs['std'] = std
        self.ds_test.standardize(**kwargs)

    def minmax_scale(self, **kwargs):
        min, max = self.ds_train.minmax_scale(**kwargs)
        
        kwargs['min'] = min
        kwargs['max'] = max
        self.ds_test.minmax_scale(**kwargs)

    def feature_selection_by(self, *args, **kwargs):
        kwargs['indices'] = self.ds_train.feature_selection_by(*args, **kwargs)
        self.ds_test.feature_selection_by(*args, **kwargs)

    def SMOTE(self, *args, **kwargs):
        self.ds_train.SMOTE(*args, **kwargs)

    def train_val_test_split(self, **kwargs):
        return self.ds_train, self.ds_test

    def __len__(self):
        return len(self.ds_train) + len(self.ds_test)

    def __getitem__(self, idx):
        # Does not support slices
        return self.ds_train[idx] if idx < len(self.ds_train) \
               else self.ds_test[idx - len(self.ds_train)]
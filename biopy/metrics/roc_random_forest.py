import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import torch
from torch.utils import data

from collections import defaultdict

from ..models import DomainTranslator
from ..datasets.datasets_nature import NucleiDatasetWrapper, OmicDatasetWrapper


class ROCRandomForest:
    supports_train = True
    separate_omics = False

    def __init__(self, dataset, aaes, device='cuda', hparams=None, **kwargs):
        
        self.dataset = dataset
        self.aaes = aaes
        self.device = device
        
        self.translators = defaultdict(lambda: {})
        for oid1 in self.dataset.keys():
            for oid2 in self.dataset.keys():
                self.translators[oid1][oid2] = DomainTranslator(aaes[oid1].encoder, aaes[oid2].decoder).to(
                    self.device).eval()
        
        # Setting default values for the classifiers
        hparams = {} if hparams is None else hparams.copy()
        hparams['max_depth'] = kwargs.get('max_depth', 2)
        hparams['n_estimators'] = kwargs.get('n_estimators', 100)
        self.clfs = {omic: RandomForestClassifier(**hparams) for omic in self.dataset.keys()}
        for omic in self.clfs:
            if isinstance(dataset[omic], NucleiDatasetWrapper) or isinstance(dataset[omic], OmicDatasetWrapper) or hasattr(dataset[omic], 'no_slicing'):
                data = np.array(list(map(lambda x: x[0].flatten().numpy(), dataset[omic])))
                labels = np.array(list(map(lambda x: x[1], dataset[omic])))
                self.clfs[omic].fit(data, labels)
            else:
                self.clfs[omic].fit(dataset[omic][:][0].cpu().numpy(), dataset[omic][:][1].cpu().numpy())
            

    def __call__(self, omics_in=None, omics_out=None, mean_strategy='only_translations', **kwargs):
        if omics_in is not None:
            assert omics_out is not None, "Must set both omics_in and omics_out if omics_in is set"
            return self._get_error(omics_in, omics_out)
        else:
            if len(list(self.dataset.keys())) == 1:
                omic = list(self.dataset.keys())[0]
                omics_pairs = [(omic, omic)]
            elif mean_strategy == 'only_translations':
                #assert len(list(self.dataset.keys())) > 1, "'only_translations' strategy can be used if dataset contains more than 1 omic"
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in self.dataset.keys() \
                                if omic_src != omic_dst]
            else:
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in self.dataset.keys()]

            return sum([self._get_error(*omics_pair) for omics_pair in omics_pairs]) / len(omics_pairs)

    def _get_error(self, omics_in, omics_out):
        dataloader = data.DataLoader(self.dataset[omics_in], batch_size=32, shuffle=False, num_workers=1, 
                                     pin_memory=True, drop_last=False)
        
        translations = []
        for batch_features, _ in dataloader:
            res = self.translators[omics_in][omics_out](batch_features.to(self.device))
            if omics_out == 'nuclei-images':
                translations.append(res.detach().view(res.shape[0], -1).cpu())                
            else:
                translations.append(res.detach().cpu())

        translations = torch.cat(translations, dim=0).numpy()
        if len(self.clfs[omics_out].classes_) > 2:
            # probability_class_i = votes_class_i / total_votes
            y_pred = self.clfs[omics_out].predict_proba(translations)
        else:
            # In the binary case, `roc_auc_score` does not want class probabilities
            y_pred = self.clfs[omics_out].predict(translations)
        
        if isinstance(self.dataset[omics_in], NucleiDatasetWrapper) or isinstance(self.dataset[omics_in], OmicDatasetWrapper) or hasattr(self.dataset[omics_in], 'no_slicing'):
            y_true = np.array(list(map(lambda x: x[1], self.dataset[omics_in])))
        else:
            y_true = self.dataset[omics_in][:][1].cpu().numpy()
        
        return roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro')

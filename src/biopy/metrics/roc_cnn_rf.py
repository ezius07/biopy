import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils import data

import os

from collections import defaultdict

from ..datasets.datasets_nature import NucleiDatasetWrapper, OmicDatasetWrapper
from ..models import DomainTranslator


def train_cnn(dataset, device):
    cached_model = f'.cnn'

    train_dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)

    net = nn.Sequential(*[
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, padding=0),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, padding=0),
        nn.Flatten(1),
        nn.Linear(16 * 16 * 64, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ])

    if os.path.isfile(cached_model):
        model = torch.load(cached_model)
        net.load_state_dict(model)
        net.eval()
        return net

    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss()

    net.to(device)
    for _ in range(20):
        net.train()
        for X_batch, Y_batch in train_dl:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            criterion(net(X_batch), Y_batch).backward()
            optimizer.step()
    net.eval()
    # this network is trained on the SMOTED version of the train split of images data

    torch.save(net.state_dict(), cached_model)
    return net


@torch.no_grad()
def get_preds(dati, net, device):
    net.eval()
    net.to(device)

    dati = np.reshape(dati, (dati.shape[0], 1, 64, 64))
    
    dataset = torch.tensor(dati)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, pin_memory=True)
    
    output_list = []
    for batch in train_dl:
        outputs = net(batch.to(device))
        output_list.append(outputs.detach().cpu())

    outputs = torch.cat(output_list, 0)
    _, predicted = torch.max(outputs, 1)
    return predicted.cpu()


def train_rf(dataset, hparams=None):
    hparams = {} if hparams is None else hparams.copy()
    hparams['max_depth'] = hparams.get('max_depth', 2)
    hparams['n_estimators'] = hparams.get('n_estimators', 100)
    if isinstance(dataset, NucleiDatasetWrapper) or isinstance(dataset, OmicDatasetWrapper) or hasattr(
            dataset, 'no_slicing'):
        data = np.array(list(map(lambda x: x[0].flatten().numpy(), dataset)))
        labels = np.array(list(map(lambda x: x[1], dataset)))
        return RandomForestClassifier(**hparams).fit(data, labels)
    else:
        return RandomForestClassifier(**hparams).fit(dataset[:][0].cpu().numpy(), dataset[:][1].cpu().numpy())


class ROCCNNRF:
    supports_train = True
    separate_omics = False

    def __init__(self, dataset, aaes, device='cuda', **kwargs):
        self.dataset = dataset
        self.aaes = aaes
        self.device = device

        self.translators = defaultdict(lambda: {})
        for oid1 in self.dataset.keys():
            for oid2 in self.dataset.keys():
                self.translators[oid1][oid2] = DomainTranslator(aaes[oid1].encoder, aaes[oid2].decoder).to(
                    self.device).eval()

        self.clfs = {
            omic: (
                train_cnn(self.dataset[omic], self.device) if omic == 'nuclei-images' else train_rf(self.dataset[omic])) \
            for omic in self.dataset.keys()
        }

    def __call__(self, omics_in=None, omics_out=None, mean_strategy='only_translations', **kwargs):
        if omics_in is not None:
            assert omics_out is not None, "Must set both omics_in and omics_out if omics_in is set"
            return self._get_error(omics_in, omics_out)
        else:
            if len(list(self.dataset.keys())) == 1:
                omic = list(self.dataset.keys())[0]
                omics_pairs = [(omic, omic)]
            elif mean_strategy == 'only_translations':
                # assert len(list(self.dataset.keys())) > 1, "'only_translations' strategy can be used if dataset contains more than 1 omic"
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in
                               self.dataset.keys() \
                               if omic_src != omic_dst]
            else:
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in
                               self.dataset.keys()]

            return sum([self._get_error(*omics_pair) for omics_pair in omics_pairs]) / len(omics_pairs)

    def _get_error(self, omics_in, omics_out):
        # Get translations
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
        if omics_out != "nuclei-images":
            if len(self.clfs[omics_out].classes_) > 2:
                # probability_class_i = votes_class_i / total_votes
                y_pred = self.clfs[omics_out].predict_proba(translations)
            else:
                # In the binary case, `roc_auc_score` does not want class probabilities
                y_pred = self.clfs[omics_out].predict(translations)
        else:
            # TODO: there is no predict_proba for the CNN yet
            # (ok so far because the CNN is used only on the CD4 dataset which has binary labels)
            y_pred = get_preds(translations, self.clfs[omics_out], self.device)
        
        y_true = np.array(list(map(lambda x: x[1], self.dataset[omics_in])))

        return roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro')

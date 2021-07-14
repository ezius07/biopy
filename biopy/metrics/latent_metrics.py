import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import torch
from torch.utils import data


class KNNAccuracySklearn:
    supports_train = True
    separate_omics = False

    def __init__(self, dataset, aaes, device='cuda', k=10, **kwargs):
        self.dataset = dataset
        self.aaes = aaes
        self.device = device
        self.latents = {omic: self._get_latents(omic) for omic in self.dataset.keys()}
        # metric='minkowski', p=1 -> l1 norm
        self.nbrs = {omic: NearestNeighbors(n_neighbors=k, metric='minkowski', p=1).fit(self.latents[omic]) for omic in
                     self.dataset.keys()}

    def _get_latents(self, omic):
        dataloader = data.DataLoader(self.dataset[omic], batch_size=32, shuffle=False,
                                     pin_memory=True, drop_last=False)

        latents = []
        with torch.no_grad():
            for batch_features, _ in dataloader:
                res = self.aaes[omic].to(self.device).encode_and_sample(batch_features.to(self.device))
                latents.append(res)
        latents = torch.cat(latents, dim=0).detach().cpu().numpy()
        return latents

    def __call__(self, omics_in=None, omics_out=None, mean_strategy='only_translations', **kwargs):
        if omics_in is not None:
            assert omics_out is not None, "Must set both omics_in and omics_out if omics_in is set"
            return self._get_error(omics_in, omics_out)
        else:
            if mean_strategy == 'only_translations':
                assert len(list(
                    self.dataset.keys())) > 1, "'only_translations' strategy can be used if dataset contains more than 1 omic"
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in
                               self.dataset.keys() \
                               if omic_src != omic_dst]
            else:
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in
                               self.dataset.keys()]

            return sum([self._get_error(*omics_pair) for omics_pair in omics_pairs]) / len(omics_pairs)

    def _get_error(self, omic_a, omic_b):
        indices = self.nbrs[omic_a].kneighbors(self.latents[omic_b], return_distance=False)

        sample_is_in_nn = (indices == np.arange(indices.shape[0])[:, np.newaxis]).any(
            axis=1)  # returns [True, False, False, ....]

        return sample_is_in_nn.mean()  # Same as summing al trues and then dividing by n


class FractionCloserSklearn(KNNAccuracySklearn):
    supports_train = True
    separate_omics = False

    def __init__(self, dataset, aaes, device='cuda', **kwargs):
        super(FractionCloserSklearn, self).__init__(dataset, aaes, device, k=1)

    def _get_error(self, omic_a, omic_b):
        latents_a = self.latents[omic_a]
        latents_b = self.latents[omic_b]

        distances_all_points, _ = self.nbrs[omic_a].kneighbors(latents_b, n_neighbors=latents_b.shape[0],
                                                               return_distance=True)  # Euclidean distances are returned
        distance_true_nn = np.linalg.norm((latents_a - latents_b), axis=1, ord=1)  # l1 norm

        n_closer = (distances_all_points <= distance_true_nn[:, np.newaxis]).sum(axis=1) - 1
        fractions_closer = (n_closer / distances_all_points.shape[0])

        return 1 - fractions_closer.mean()


class FractionCorrectCluster:
    supports_train = False
    separate_omics = True

    def __init__(self, dataset, aaes, device='cuda', **kwargs):
        self.dataset_metric = dataset
        self.aaes = aaes
        self.device = device
        assert "train_dataset" in kwargs
        self.dataset = kwargs["train_dataset"]

    def _get_latents(self, omic, dataset):
        dataloader = data.DataLoader(dataset[omic], batch_size=32, shuffle=False,
                                     pin_memory=True, drop_last=False)

        latents = []
        with torch.no_grad():
            for batch_features, _ in dataloader:
                res = self.aaes[omic].encode_and_sample(batch_features.to(self.device))
                latents.append(res)
        latents = torch.cat(latents, dim=0).detach().cpu().numpy()
        return latents

    @ignore_warnings(category=ConvergenceWarning)
    def __call__(self, omics_train=None, omics_test=None, mean_strategy='only_translations', **kwargs):
        if omics_train is not None:
            assert omics_test is not None, "Must set both omics_in and omics_out if omics_in is set"
            if not isinstance(omics_train, list):
                omics_train = [omics_train]
            if not isinstance(omics_test, list):
                omics_test = [omics_test]
            self.latents = np.concatenate(
                list({omic: self._get_latents(omic, self.dataset) for omic in omics_train}.values()))
            self.labels = np.concatenate([np.array([self.dataset[omic][x][1] for x in range(len(self.dataset[omic]))]) for omic in omics_train])

            self.classifier = LogisticRegression().fit(self.latents, self.labels)
            self.latents_metric = np.concatenate(list(
                {omic: self._get_latents(omic, self.dataset_metric) for omic in omics_test}.values()))
            self.labels_metric = np.concatenate(
                [self.dataset_metric[omic].labels for omic in omics_test])
        else:
            self.latents = np.concatenate(
                list({omic: self._get_latents(omic, self.dataset) for omic in self.dataset.keys()}.values()))
            self.labels = np.concatenate([self.dataset[omic].labels for omic in self.dataset.keys()])
            self.classifier = LogisticRegression().fit(self.latents, self.labels)
            self.latents_metric = np.concatenate(list(
                {omic: self._get_latents(omic, self.dataset_metric) for omic in self.dataset_metric.keys()}.values()))
            self.labels_metric = np.concatenate(
                [self.dataset_metric[omic].labels for omic in self.dataset_metric.keys()])

        predicted = self.classifier.predict(self.latents_metric)
        return ((predicted == self.labels_metric).sum()) / len(self.latents_metric)

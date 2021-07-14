import torch
import torch.nn as nn
from torch.utils import data
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from ..models import DomainTranslator


class LatentSpace:

    def __init__(self, dim, model, dataset, omics=['miRNA', 'mRNA', 'meth27-450-preprocessed']):

        self.dim = dim
        self.omics = omics
        self.num_omics = 0
        self.latent_mapping = {}
        self.pointsByOmicsID = {}

        dataset.standardize()
        _, dtest = dataset.train_val_test_split()

        for omic in omics:
            self.__gen_latent_space(model, dtest, omic)

        self.__genPointsByOmicsID()

    def __gen_latent_space(self, model, dataset, omic):
        '''creates dict latent_mapping with associations
        key:encoded_point -> val:original_data_point
        NOTE: does not work with images'''
        dataset = dataset.set_omic(omic)
        self.num_omics += len(dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batches = 32
        loader = data.DataLoader(dataset, batch_size=batches, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)
        model.eval()
        model.to(device)
        for batch_features, _ in loader:
            batch_cuda = batch_features.to(device)

            z = model.encode_and_sample(batch_cuda)
            z = z.detach().cpu()

            for i, data_point in enumerate(z):
                self.latent_mapping[data_point] = (batch_features[i], omic)

    def __euclidean_distance(self, input, target):
        return torch.cdist(input.unsqueeze(0), target.unsqueeze(0))[0]

    def __genPointsByOmicsID(self):
        for omic in self.omics:
            self.pointsByOmicsID[omic] = [k for k, v in self.latent_mapping.items() if v[1] == omic]

    def __findNearestNeighbor(self, sample, target_samples):

        min_dist = np.inf
        ts_min_dist = None

        for ts in target_samples:
            dist = self.__euclidean_distance(sample, ts)
            if dist < min_dist:
                min_dist = dist
                ts_min_dist = ts

        return min_dist, ts_min_dist

    def nearestNeighbor(self, sample, target_omics_id):
        target_samples = self.pointsByOmicsID[omic_id]  # filter by omics id
        distance, ts_min_dist = self.__findNearestNeighbor(sample, target_samples)
        return distance, ts_min_dist

    def check_content(self):
        for k, v in self.latent_mapping:
            print(f"({k}, {v})")

    def __len__(self):
        return len(self.latent_mapping)


class ReconstructionError:
    """
    decoders: ordered list of decoders (by omics id)
    latent_space: LatentSpace object
    """

    def __init__(self, decoders, latent_space):
        self.decoders = decoders
        self.latent_space = latent_space

    def pointwise_error(self, **kwargs):

        # input: exit of a decoder. target: sample to compare with
        if 'input' in kwargs.keys() and 'target' in kwargs.keys():

            input, target = kwargs['input'], kwargs['target']

            return torch.cdist(input.unsqueeze(0), target.unsqueeze(0))[0]

        # input_latent: point in the latent space (same object)
        # input_omics_id: omics_id of input_latent
        # target_omics_id: omics_id of the samples to evaluate the pointwise error with
        elif 'input_latent' in kwargs.keys() and \
                'input_omics_id' in kwargs.keys() and \
                'target_omics_id' in kwargs.keys():

            input_latent = kwargs['input_latent']
            io_id = kwargs['input_omics_id']
            to_id = kwargs['target_omics_id']

            input = self.decoders[io_id](input_latent)  # get the exit of the proper decoder

            # get the nearest target sample of the correct target omics id
            _, ts_min_dist = self.latent_space.nearestNeighbor(input_latent, to_id)

            target = self.latent_space.latent_mapping[ts_min_dist][0]  # find the final target

            return torch.cdist(input.unsqueeze(0), target.unsqueeze(0))[0] / (distance + 1)

        else:
            raise Exception("Not handled")
            return None

    def class_error(self, io_id, to_id):

        io_samples = self.latent_space.filterByOmicsID(io_id)

        cum_error = 0
        for input_latent in io_sample:
            cum_error += self.pointwise_error(
                input_latent=input_latent,
                input_omics_id=io_id,
                target_omics_id=to_id
            )

        return cum_error / len(io_samples)

    def cumulative_error(self):

        cum_error = 0

        for io_id in range(self.latent_space.num_omics):
            for to_id in range(self.latent_space.num_omics):
                cum_error += self.class_error(io_id, to_id)

        return cum_error / (self.latent_space.num_omics ** 2)


"""
HOW TO USE:
ore = OmicsReconstructionError(dataset_test_omics, aaes)
ore('mRNA', 'miRNA')
"""


class OmicsReconstructionError:
    """
    dataset: a dictionary of the form: 
    {
        'mRNA': Dataset,
        'miRNA': Dataset,
        'meth27-450-preprocessed': Dataset
    }
    aaes: a dictionary of the form:
    {
        'mRNA': AAE,
        'miRNA': AAE,
        'meth27-450-preprocessed': AAE
    }
    """
    supports_train = False
    separate_omics = False

    def __init__(self, dataset, aaes, device, **kwargs):
        self.dataset = dataset
        self.device = device
        self.translators = defaultdict(lambda: {})
        for oid1 in self.dataset.keys():
            for oid2 in self.dataset.keys():
                self.translators[oid1][oid2] = DomainTranslator(aaes[oid1].encoder, aaes[oid2].decoder).to(
                    self.device).eval()

    """
    omics_in: id omics that is autoencoded
    omics_out: id omics to calculate the error with
    return: MSE between the two datasets of the corresponding omics
    """

    def __call__(self, omics_in=None, omics_out=None, mean_strategy='only_translations', **kwargs):
        if omics_in is not None:
            assert omics_out is not None, "Must set both omics_in and omics_out if omics_in is set"
            return self._get_error(omics_in, omics_out)
        else:
            if mean_strategy == 'only_translations':
                assert len(list(self.dataset.keys())) > 1, "'only_translations' strategy can be used if dataset contains more than 1 omic"
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in self.dataset.keys() \
                                if omic_src != omic_dst]
            else:
                omics_pairs = [(omic_src, omic_dst) for omic_src in self.dataset.keys() for omic_dst in self.dataset.keys()]

            return sum([self._get_error(*omics_pair) for omics_pair in omics_pairs]) / len(omics_pairs)

    def _get_error(self, omics_in, omics_out):
        cum_err = 0
        for index in range(len(self.dataset[omics_in])):
            cum_err += self._error_sample(omics_in, omics_out, index)
        return (cum_err / len(self.dataset[omics_in])).item()

    def _error_sample(self, omics_in, omics_out, index):
        return MSELoss(reduction='mean')(
            self.translators[omics_in][omics_out](self.dataset[omics_in][index][0].unsqueeze(0).to(self.device)),
            self.dataset[omics_out][index][0].unsqueeze(0).to(self.device))

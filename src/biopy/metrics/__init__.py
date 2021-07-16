__all__ = ['reconstruction_error', 'roc_random_forest', 'paper_metrics']

from .reconstruction_error import LatentSpace, ReconstructionError, OmicsReconstructionError
from .roc_random_forest import ROCRandomForest
from .roc_cnn_rf import ROCCNNRF, train_cnn
from .latent_metrics import KNNAccuracySklearn, FractionCloserSklearn, FractionCorrectCluster

def print_all_conversion_errors(error_class, trainer, omics=["mRNA", "miRNA", "meth27-450-preprocessed"], **kwargs):

    """
    @param error_class: the class of the metric which wants to be evaluated
    @param trainer: a dictionary containing as key the omics and as value the network trained on the omic
    @param omics: the omics on which you want to evaluate the conversion
    @param kwargs: additional arguments for the error_class
    """
    for omic in omics:
        assert omic in trainer.keys(), "One of the omics for the evaluation is not in the trainer dictionary"

    dataset = {omic: trainer[omic].test_dataset for omic in omics}
    aaes = {omic: trainer[omic].model for omic in omics}
    error = error_class(dataset, aaes, trainer[omics[0]].device, **kwargs)
    for omic in omics:
        for omic2 in omics:
            print("Conversion from {} to {}".format(omic, omic2))
            print(error(omic, omic2))

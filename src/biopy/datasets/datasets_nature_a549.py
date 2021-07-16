from .datasets_gdc import DatasetMultiOmicsGDC, DatasetMultiOmicsGDCTrainTest

class DatasetMultiOmicsNatureA549(DatasetMultiOmicsGDC):

    def load_omics_from_disk(self, folder='dataset_nature_atac-rna', omics=('rna', 'atac'), labels='labels', labels_columns=["treatment_time"], **kwargs):
        super().load_omics_from_disk(folder=folder, omics=omics, labels=labels, labels_columns=labels_columns, **kwargs)
    
    def train_val_test_split(self, test=.2, validation=0, random_state=97):
        return super(DatasetMultiOmicsNatureA549, self).train_val_test_split(test=test, validation=validation, random_state=97)
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, entity_features, labels):
        """
        entity_features: FloatTensor with size (n_entities, n_features)
        labels: FloatTensor with size (x,3), where x is the number of labeled combinations
                this Tensor has the column format [entity1_index, entity2_index, y]
        """


    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return 

    def __getitem__(self, idx):
        """
        returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_features) corresponding to entity1
            2. FloatTensor with size (n, n_features) corresponding to entity2
            3. FloatTensor with size (n, 1) corresponding to the actual label 
        """

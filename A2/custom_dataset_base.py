import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, entity_features, labels):
        """
        Initialization of the dataset.

        entity_features: FloatTensor with size (n_entities, n_features)
        labels: FloatTensor with size (x,3), where x is the number of labeled combinations
                this Tensor has the column format [entity1_index, entity2_index, y]

        Space complexity: O(n^2 + nk) where n is the number of entities and k is the number of features
        """
        self.entity_features = entity_features
        self.labels = labels
        self.n_features = entity_features.shape[1]

    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(labels)

    def __getitem__(self, idx : int):
        """
        Returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_features) corresponding to entity1
            2. FloatTensor with size (n, n_features) corresponding to entity2
            3. FloatTensor with size (n, 1) corresponding to the actual label 
        
        Method to support indexing such that dataset[i] can be used to get ith sample.
        """
        label = self.labels[idx]

        return (torch.reshape(self.entity_features[label[0].int().item()], (-1, self.n_features)),
                torch.reshape(self.entity_features[label[1].int().item()], (-1, self.n_features)),
                torch.reshape(label[2], (-1, 1)))
        
entity_features = pd.read_csv('dummy_features.csv', header=None)
labels = pd.read_csv('dummy_data.csv')
dataset = CustomDataset(
    torch.Tensor(entity_features.to_numpy()), 
    torch.Tensor(labels.to_numpy())
    )

# using a dataloader to load the data
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for i,data in enumerate(dataloader):
    (x1,x2,y) = data
    print("yo")
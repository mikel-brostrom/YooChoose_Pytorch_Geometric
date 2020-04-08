import torch
from torch_geometric.data import Dataset


# The class inherits the base class Dataset from pytorch
class LoadData(Dataset):  # for training/testing
    def __init__(self, data_path):
        super(LoadData, self).__init__()
        self.data = torch.load(data_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

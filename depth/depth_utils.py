from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class DepthBase(ABC, Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    @abstractmethod
    def load_image(self, filename):
        pass

    @abstractmethod
    def load_depth(self, filename):
        pass

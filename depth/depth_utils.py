from torch.utils.data import Dataset
from abc import abstractmethod
import h5py
import os

from .. import dataset_utils

module_path = os.path.dirname(os.path.realpath(__file__))

class DepthBase(Dataset):

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


class DepthHDF5(dataset_utils.HDF5Dataset):

    def base_path(self):
        return module_path

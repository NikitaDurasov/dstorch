from torch.utils.data import Dataset
from abc import abstractmethod
import h5py
import os

from .. import dataset_utils

module_path = os.path.dirname(os.path.realpath(__file__))

class DepthBase(dataset_utils.DatasetBase):

    def load_image(self, filename):
        pass

    def load_depth(self, filename):
        pass

    def generate_load_functions(self):
        return [self.load_image, self.load_depth]

    # TODO GET RID OF THIS 
    def base_path(self):
        return module_path

    def keys(self):
        return ['image', 'depth']


class DepthHDF5(dataset_utils.HDF5Dataset):

    def base_path(self):
        return module_path

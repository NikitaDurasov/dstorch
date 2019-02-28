from torch.utils.data import Dataset
from abc import abstractmethod
from .. import dataset_utils
import h5py
import os

module_path = os.path.dirname(os.path.realpath(__file__))

class SegmentationBase(dataset_utils.DatasetBase):

    def load_image(self, filename):
        pass

    def load_segmentation(self, filename):
        pass

    def generate_load_functions(self):
        return [self.load_image, self.load_segmentation]

    # TODO GET RID OF THIS 
    def base_path(self):
        return module_path

    def keys(self):
        return ['image', 'mask']
        

class SegmentationHDF5(dataset_utils.HDF5Dataset):

    def base_path(self):
        return module_path
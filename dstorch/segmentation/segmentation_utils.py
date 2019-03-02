from torch.utils.data import Dataset
from abc import abstractmethod
import h5py
import os

from .. import dataset_utils


class SegmentationBase(dataset_utils.DatasetBase):

    def load_image(self, filename):
        pass

    def load_segmentation(self, filename):
        pass

    def generate_load_functions(self):
        return [self.load_image, self.load_segmentation]

    # TODO GET RID OF THIS 
    def base_path(self):
        return "segmentation"

    def keys(self):
        return ['image', 'mask']
        

class SegmentationHDF5(dataset_utils.HDF5Dataset):

    def base_path(self):
        return "segmentation"

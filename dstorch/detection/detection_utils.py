from torch.utils.data import Dataset
from abc import abstractmethod
import h5py
import os

from .. import dataset_utils


class DetectionBase(dataset_utils.DatasetBase):

    def load_image(self, filename):
        pass

    def load_bbox(self, filename):
        pass

    def generate_load_functions(self):
        return [self.load_image, self.load_bbox]

    # TODO GET RID OF THIS 
    def base_path(self):
        return "detection"

    def keys(self):
        return ['image', 'bbox']
    
class LabeledDetectionBase(dataset_utils.DatasetBase):

    def load_image(self, filename):
        pass

    def load_bbox(self, filename):
        pass
    
    def load_label(self, filename):
        pass

    def generate_load_functions(self):
        return [self.load_image, self.load_bbox, self.load_label]

    # TODO GET RID OF THIS 
    def base_path(self):
        return "detection"

    def keys(self):
        return ['image', 'bbox', 'label']

class DetectionHDF5(dataset_utils.HDF5Dataset):

    def base_path(self):
        return "detection"
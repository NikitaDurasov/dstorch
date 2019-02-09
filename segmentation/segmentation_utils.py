from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class SegmentationBase(ABC, Dataset):

    def __init__(self, splits_file, transform):
        self.splits_file = splits_file
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    @abstractmethod
    def load_image(self, filename):
        pass

    @abstractmethod
    def load_segmentation(self, filename):
        pass

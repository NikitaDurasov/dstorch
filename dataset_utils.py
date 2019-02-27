from torch.utils.data import Dataset
from abc import abstractmethod
import h5py
import os

class DatasetBase(Dataset):

    def __init__(self, data_path, split, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.split = split

        self.data_files = self.generate_split()
        self.load_functions = self.generate_load_functions()

    def __len__(self):
        return len(self.data_files)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def generate_load_functions(self):
        pass

    def keys(self):
        pass

    def __getitem__(self, index):
        sample = self.data_files[index]
        sample = {key: load(filename) for load, filename, key in zip(self.load_functions, sample, self.keys())}

        if self.transform:
            sample = self.transform(**sample)

        return sample

    def line_to_filepaths(self, line):
        line = line.rstrip()
        file_names = line.split("\t")

        sample_files = []

        for name in file_names:
            split_name = name.split(" ")

            if split_name[0] == "-":
                sample_files.append("-")
            else:
                sample_files.append(os.path.join(*([self.data_path] + split_name)))

        return sample_files

    # TODO GET RID OF THIS 
    @abstractmethod
    def base_path(self):
        pass

    def generate_split(self):

        if self.split == "train":
            split_path = os.path.join(self.base_path(), "splits", self.name() + "_train_split.txt")
            split_file = open(split_path)
            return list(map(self.line_to_filepaths, split_file.readlines()))

        elif self.split == "valid":
            split_path = os.path.join(self.base_path(), "splits", self.name() + "_val_split.txt")
            split_file = open(split_path)
            return list(map(self.line_to_filepaths, split_file.readlines()))

        elif self.split == "test":
            split_path = os.path.join(self.base_path(), "splits", self.name() + "_test_split.txt")
            split_file = open(split_path)
            return list(map(self.line_to_filepaths, split_file.readlines()))

class HDF5Dataset(Dataset):

    def __init__(self, hdf5_path, split, transform=None):
        self.data_path = hdf5_path
        self.transform = transform
        self.split = split

        hdf5_data = h5py.File(self.data_path, 'r')
        self.indexes = self.generate_split()

        self.keys = self.generate_keys()
        self.data = [hdf5_data.get(key) for key in self.keys]
        self.new_keys = self.generate_new_keys()

    def __len__(self):
        return len(self.indexes)

    @abstractmethod
    def generate_keys(self):
        pass

    @abstractmethod
    def generate_new_keys(self):
        pass

    @abstractmethod
    def process_data(self, sample):
        pass

    @abstractmethod
    def name(self):
        pass

    # TODO GET RID OF THIS 
    @abstractmethod
    def base_path(self):
        pass

    def __getitem__(self, index):
        sample = {}
        idx = self.indexes[index]

        for key, data in zip(self.new_keys, self.data):
            sample[key] = data[idx]

        sample = self.process_data(sample)

        if self.transform:
            sample = self.transform(**sample)

        return sample

    def generate_split(self):

        if self.split == 'train':
            indexes_path = os.path.join(self.base_path(), "splits", self.name() + "_train_split.txt")
            indexes_file = open(indexes_path)
            return list(map(int, indexes_file.readlines()))

        if self.split == 'valid':
            indexes_path = os.path.join(self.base_path(), "splits", self.name() + "_val_split.txt")
            indexes_file = open(indexes_path)
            return list(map(int, indexes_file.readlines()))

        if self.split == 'test':
            indexes_path = os.path.join(self.base_path(), "splits", self.name() + "_test_split.txt")
            indexes_file = open(indexes_path)
            return list(map(int, indexes_file.readlines()))

from torch.utils.data import Dataset
from abc import abstractmethod


class SegmentationBase(Dataset):

    def __init__(self, data_path, transform, split):
        self.data_path = data_path
        self.transform = transform
        self.split = split

        self.data_files = self.generate_split(data_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        sample = self.data_files[index]

        image_name = sample[0]
        mask_name = sample[1]

        image = self.load_image(image_name)
        mask = self.load_segmentation(mask_name)

        sample = {"image": image, "mask": mask}

        if self.transform:
            sample = self.transform(**sample)

        return sample

    @abstractmethod
    def load_image(self, filename):
        pass

    @abstractmethod
    def load_segmentation(self, filename):
        pass

    @abstractmethod
    def generate_split(self, data_path):
        pass

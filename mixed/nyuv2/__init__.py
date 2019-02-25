from ... import dataset_utils
import os

module_path = os.path.dirname(os.path.realpath(__file__))

class NYUv2DepthAndSegmentation(dataset_utils.HDF5Dataset):

    def generate_keys(self):
        return ["images", "labels", "depths"]

    def generate_new_keys(self):
        return ["image", "mask", "depth"]

    def name(self):
        return "nyuv2"

    def process_data(self, sample):
        sample['image'] = sample['image'].swapaxes(0, 2)
        sample['mask'] = sample['mask'].swapaxes(0, 1)
        sample['depth'] = sample['depth'].swapaxes(0, 1)

        return sample

    def base_path(self):
        return os.path.dirname(module_path)

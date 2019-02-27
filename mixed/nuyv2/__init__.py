from ... import dataset_utils

class NYUv2DepthAndSegmentation(dataset_utils.HDF5Dataset):

    def generate_keys(self):
        return ["images", "labels", "depths"]

    def generate_new_keys(self):
        return ["image", "mask", "depths"]

    def name(self):
        return "nuyv2"

    def process_data(self, sample):
        sample['image'] = sample['image'].swapaxes(0, 2)
        sample['mask'] = sample['mask'].swapaxes(0, 1)
        sample['depth'] = sample['depth'].swapaxes(0, 1)

        return sample
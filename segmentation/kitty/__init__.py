from .. import segmentation_utils
from PIL import Image
import numpy as np
import os


class KittyDataset(segmentation_utils.SegmentationBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_segmentation(self, filename):
        if filename == '-':
            return None
        mask = Image.open(filename)
        return np.array(mask)

    def generate_split(self, data_path):

        if self.split == "train":
            train_path = os.path.join(data_path, "training")
            image_path = os.path.join(train_path, "image_2")
            mask_path = os.path.join(train_path, "semantic")

            images = os.listdir(image_path)

            masks = [os.path.join(mask_path, file) for file in images]
            images = [os.path.join(image_path, file) for file in images]

            return list(zip(images, masks))

        elif self.split == "test":
            train_path = os.path.join(data_path, "testing")
            image_path = os.path.join(train_path, "image_2")

            images = os.listdir(image_path)
            images = [os.path.join(image_path, file) for file in images]
            masks = ["-"] * len(images)

            return list(zip(images, masks))

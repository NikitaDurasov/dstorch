from .. import segmentation_utils
from PIL import Image
import numpy as np
import os
import sys

class KittyDataset(segmentation_utils.SegmentationBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_segmentation(self, filename):
        if filename == '-':
            return None
        mask = Image.open(filename)
        return np.array(mask)

    def name(self):
        return "kitty"

from .. import depth_utils
from PIL import Image
import numpy as np
import os
import glob


class KittyDataset(depth_utils.DepthBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_depth(self, filename):
        if filename == '-':
            return None
        mask = Image.open(filename)
        return np.array(mask)

    def name(self):
        return "kitty"
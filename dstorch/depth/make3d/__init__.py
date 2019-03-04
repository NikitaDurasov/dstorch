from .. import depth_utils
from PIL import Image
import numpy as np
import os
import glob

from scipy.io import loadmat
import scipy
import scipy.misc


class Make3dDataset(depth_utils.DepthBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_depth(self, filename):
        if filename == '-':
            return None
        depth = loadmat(filename)['Position3DGrid']
        depth = scipy.misc.imresize(depth, (2272, 1704))
        return np.array(depth)

    def name(self):
        return "make3d"
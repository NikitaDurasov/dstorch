from .. import segmentation_utils
from PIL import Image
import numpy as np

class PascalVOC2012Dataset(segmentation_utils.SegmentationBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_segmentation(self, filename):
        if filename == '-':
            return None
        mask = Image.open(filename)
        mask = np.array(mask)

        # TODO add borders option
        mask[mask == 255] = 0
        return mask

    def name(self):
        return "voc2012"
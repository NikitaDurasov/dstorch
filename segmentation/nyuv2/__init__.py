from .. import segmentation_utils
from PIL import Image
import numpy as np

class NYUv2Segmentation(segmentation_utils.SegmentationHDF5):

	def generate_keys(self):
		return ["images", "labels"]

	def generate_new_keys(self):
		return ["image", "mask"]

	def name(self):
		return "nyuv2"

	def process_data(self, sample):
		sample['image'] = sample['image'].swapaxes(0, 2)
		sample['mask'] = sample['mask'].swapaxes(0, 1)

		return sample

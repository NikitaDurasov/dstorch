from .. import depth_utils
from PIL import Image
import numpy as np

class NYUv2Depth(depth_utils.DepthHDF5):

	def generate_keys(self):
		return ["images", "depths"]

	def generate_new_keys(self):
		return ["image", "depth"]

	def name(self):
		return "nyuv2"

	def process_data(self, sample):
		sample['image'] = sample['image'].swapaxes(0, 2)
		sample['depth'] = sample['depth'].swapaxes(0, 1)

		return sample
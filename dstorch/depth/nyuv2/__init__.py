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
		sample[self.new_keys[0]] = sample[self.new_keys[0]].swapaxes(0, 2)
		sample[self.new_keys[1]] = sample[self.new_keys[1]].swapaxes(0, 1)

		return sample

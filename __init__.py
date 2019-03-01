from . import segmentation
from . import depth
from . import mixed

import os 
import sys

# TODO unpack splits to dir in home
if not os.path.isdir("~/.dstorch_splits"):
	os.mkdir("~/.dstorch_splits")
from . import segmentation
from . import depth
from . import mixed

import os 
import sys
import zipfile

# unpack splits to dir in home
home_dir = os.path.expanduser("~")
splits_dir = os.path.join(home_dir, ".dstorch_splits")
if not os.path.isdir(splits_dir):
    os.makedirs(splits_dir)

zip_ref = zipfile.ZipFile(os.path.join('__data__', 'splits.zip'), 'r')
zip_ref.extractall(splits_dir)
zip_ref.close()

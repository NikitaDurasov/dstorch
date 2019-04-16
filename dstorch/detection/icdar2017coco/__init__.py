from .. import detection_utils
from PIL import Image
import numpy as np
import os
import glob
import json

from scipy.io import loadmat
import scipy
import scipy.misc

class COCOICDAR2017(detection_utils.LabeledDetectionBase):
    
    def __init__(self, data_path, split, transform=None, text=False):
        super(COCOICDAR2017, self).__init__(data_path, split, transform)
        json_file = os.path.join(self.data_path, "COCO_Text.json")
        self.coco_annotations = json.load(open(json_file))

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_bbox(self, filename):
        
        if filename == '-':
            return None
        
        index = filename.split('/')[-1]
        annotation_ids = self.coco_annotations['imgToAnns'][index]
        anns = [self.coco_annotations['anns'][str(ids)] for ids in annotation_ids]     
        return np.array([item['bbox'] for item in anns])
    
    def load_label(self, filename):
        
        if filename == '-':
            return None
        
        index = filename.split('/')[-1]
        annotation_ids = self.coco_annotations['imgToAnns'][index]
        anns = [self.coco_annotations['anns'][str(ids)] for ids in annotation_ids]   
        return [item['utf8_string'] if 'utf8_string' in item else None for item in anns]
    

    def name(self):
        return "icdar2017coco"
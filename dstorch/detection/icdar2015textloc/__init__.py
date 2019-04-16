from .. import detection_utils
from PIL import Image
import numpy as np
import os
import glob
import json

from scipy.io import loadmat
import scipy
import scipy.misc

import torch

class ICDAR2015TEXTLOC(detection_utils.LabeledDetectionBase):

    def load_image(self, filename):
        image = Image.open(filename)
        return np.array(image)

    def load_bbox(self, filename):
        
        if filename == '-':
            return None
        
        bboxes = open(filename, encoding='utf-8-sig').readlines()       
        bboxes_arrays = []
        for i in range(len(bboxes)):
            line = bboxes[i].strip()
            line = line.split(',')
            xy_coords = list(map(int, line[:8]))
            bboxes_arrays.append(np.array(xy_coords))
        bboxes_arrays = np.array(bboxes_arrays)
        
        return bboxes_arrays
    
    def load_label(self, filename):
        
        if filename == '-':
            return None
        
        bboxes = open(filename, encoding='utf-8').readlines()
       
        words_array = []
        for i in range(len(bboxes)):
            line = bboxes[i].strip().split(',')
            text = ','.join(line[8:])
            words_array.append(line[-1])
        
        return words_array
    
    def collate_fn(self, batch):

        images = list()
        bboxes = list()
        labels = list()

        for b in batch:
            images.append(torch.from_numpy(b['image']))
            bboxes.append(b['bbox'])
            labels.append(b['label'])

        images = torch.stack(images, dim=0)

        return {"image": images, 
                "bbox": bboxes, 
                "label": labels}


    def name(self):
        return "icdar2015textloc"
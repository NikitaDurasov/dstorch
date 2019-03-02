# dstorch: simple utilities for datasets and benchmarking 

# Getting started 

## Prerequisites

## Installation

# Usage
For the most datasets usage is rather straightforward: 
* download dataset data
* place all this data in one directory
* pass this directory path in constructor

For example let's try it with Cityscapes segmentation dataset:
1) visit [cityscapes dataset site](https://www.cityscapes-dataset.com/downloads/) and download **gtFine_trainvaltest.zip** and **leftImg8bit_trainvaltest.zip**
2) unzip this files and place them in one directory, e.g **~/Cityscapes**
```bash
Cityscapes
├── gtFine_trainvaltest
└── leftImg8bit_trainvaltest
```
3) import package and pass path to data
```python
from dstorch.segmentation import cityscapes

dataset = cityscapes.CityscapesDataset(data_path='~/Cityscapes/', split='train')

sample = dataset[0]
image, mask = sample['image'], sample['mask']

plt.imshow(image)
plt.imshow(mask, alpha=0.9)
```
![alt text](https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/stuttgart03.png)

There is also another type of datasets implemented as hdf5 storages, e.g. NYU v.2 dataset. For this type of datasets you only need to pass path to hdf5/mat file with data and that's it. 

Let's try that with NYU: 
1) download **Labeled dataset** from [site](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), now you have file named **nyu_depth_v2_labeled.mat**
2) import package and pass path to data
```python 
from dstorch.segmentation import nyuv2

dataset = nyuv2.NYUv2Segmentation('../datasets/NYUv2/nyu_depth_v2_labeled.mat', 'train')

sample = dataset[0]
image, mask = sample['image'], sample['mask']
```

## Tutorials

For detailed datasets tutorials open [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NikitaDurasov/dstorch/blob/dev/tutorials.ipynb)

Currently available:
### Segmentation
### Depth
### Mixed

## Datasets
Currently available:
### Segmentation:
* Cityscapes
* KITTY
* NYU v.2
* Pascal VOC 2012 (without test)

### Depth
* KITTY (without train)
* NYU v.2

### On the way
* Make3D depth dataset
* detections datasets

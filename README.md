# dstorch: simple utilities for datasets and benchmarking 

# Getting started 

## Prerequisites

## Installation

# Usage
For the most datasets usage is rather straightforward: 
* download dataset data
* place all this data in one directory
* pass it the path to directory to class in constructor

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


## Datasets
Currently available:
### Segmentation:
* Cityscapes
* KITTY
* NYU v.2
* Pascal VOC 2012

### Depth
* KITTY
* NYU v.2

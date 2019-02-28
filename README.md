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
```
![alt text](https://lh3.googleusercontent.com/UyhFk9TWJkGB6raXgbEOZw1uzUwW8viLMhWsOZoXgdxM1UWoP60BfnNWbccPjofbEb5Sw8YDaS17Z6SzlmK05cPwdKXAJPxBIj9eZAwb3gk2g_XIG47M_qS9VAqTViqBSWQI-2zjgjftDU5f-hP3PS7yqZZwh6Fnk_frk9ejgFcu6gh__DzZEPRB3aoO1XSuObXX4pb_HoQlNAFFSK32pV1eMHlHrJS2DWxfaewRGWAXHE5Mc9M_1OfI3gT1stEMneSTolXc3C6XdnOqsbmgB4V_Vv_rmXywioFt53z6h1E8VA8Z5OVlBqNxPsDjXScxfSLcBqT8gdPjqOOaqoovSCOogbmV5fhCNr8jttqmtfD-3KLP1kf8tJK7NrTMHRpHXqOJRSpSVeZ4OGNg-s6B0G74lo2wzUeehLor2RZ7oAjPLjMkPikNyLckzjsiIScW6CcpXy2RQLasrC01_27RB3cyzf97Jm989OPGpImmWJy2K595IKFrz6Y4xCwUMihyMz1zRX7yIePnKrSmX8sBuqm2iBMhDmwm7Ro9ZwVtg6gDw8UsK4wI-ZxS3y86JWACvQBQ43IGrJjxewWfq4iCPJ7CnZwJgCddgzgT8DagysNkzpBu4Awm96d_V_PmgxqhFOuL5g3maCMf7EGtVVP9oL7FjBBvvTs=w426-h216-no)
```python
plt.imshow(mask)
```
![alt text](https://lh3.googleusercontent.com/i7IIii_Ji2is9B0-QO2RN4uIOSjCFtVDI9svr8Nk-4BM2vgVS2T80Dt6VzKVTdPeuTxMuKM6E0xlgCb2dO5ay4QPlSdCDLMYtEON7Sh2NNJTz4RMsEra1KJIVMmKFTNo-dj4MmgEQDaobKKTCP0bblutP49xk4xeh-53qen3nO5Lb__txzfsg2ijo-eY_Ytw4cvqILYK3n6Y3XGXiYjUEDuLNGpPKjxkhU0jkoUFC9ouD2It31C6UbweR3QV6I2onoe74EpNTkZczFOK8aURwHwjDflPX8_SwWpeL_TSNaMQKRr79833CLV9Y2fzOAogLZ93n7z-Vmqj-ef2vMjoEmKog2821O8tNLPoFGrOLJUUGQOiZRtAUtTu8yLN68cnbKXJkqLihq7ur4eg6bPVba4cUfTJaC0fXX8_Ur76QS4-MDeRSvh4FIp0J5p7WE99Rylbb_DZaTlWhrk9P0bMjWVRzAlpcO8pMf83N5ibqNHFYN6JAZQo2TREFigPnNIX2b_eGnBGo8VGcaK3Y6y3WzyDQtLJsAmmnTy3sCTPs1EN_54jsFhONel3VPfkLnVl_QjztgKMlM1dYsOnqjzKyxfVOs55rDKq19Qy6jtlj90Ec62sa3XWCvPU_rQDpMzGdVHgVJngZ6yLQ1z6w0xxmICr3xncAOo=w426-h216-no)

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

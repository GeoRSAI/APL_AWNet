# AWNet
This repository is the implementation of AWNet for adaptive weighted active passive Loss. The paper "An Adaptive Weighted Method for Remote Sensing Image Retrieval with Noisy Labels" is in  https://doi.org/10.3390/app14051756.

## Data Preparing
Download dataset from the following links:
UCMD:[BaiduYun](https://pan.baidu.com/s/1JNkCT1KXQ7vtu49-tDaQFQ)
(Code: 2wah)
NWPU:[BaiduYun](https://pan.baidu.com/s/1I4EIzyXCSRqGZ7RQI8huuQ)
(Code: ep5k)
AID:[BaiduYun](https://pan.baidu.com/s/1II66-ChOuqFvXe0VCfsELQ)
(Code: pqff)

## Training and Evaluating
The pipeline for training with AWNet is the following:

 **1. Modify configuration parameters.** 

- `For example, the path of dataset: get_config("train_path", "val_path", "test_path", etc.)` 

 **2. Run the model.** 

-  `python rsir_AWNet.py`

## Contact
Please contact houdongyang1986@163.com if you have any question on the codes.

## Citation
Tian, X.; Hou, D.; Wang, S.; Liu, X.; Xing, H. An Adaptive Weighted Method for Remote Sensing Image Retrieval with Noisy Labels. Appl. Sci. 2024, 14, 1756. https://doi.org/10.3390/app14051756

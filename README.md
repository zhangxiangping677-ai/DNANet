## Dataset Introduction

NUDT-SIRST is a synthesized dataset, which contains 1327 images with resolution of 256x256. The advantage of synthesized dataset compared to real dataset lies in three aspets:

1. Accurate annotations.

2. Massive generation with low cost (i.e., time and money).

3. Numerous categories of target, rich target sizes, diverse clutter backgrounds.


## Prerequisite
* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 

* Tested on Windows 10  , with Python 3.6, PyTorch 1.1, Torchvision 0.3.0, CUDA 10.0, and 1x NVIDIA 1080Ti.

* [The NUDT-SIRST download dir](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt) (Extraction Code: nudt)

* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)

* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) [[MDvsFA]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)

## Usage

#### On windows:

```
Click on train.py and run it. 
```

nohup python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset IRSTD-1k --split_method 80_20 --model DNANet --backbone resnet_18  --deep_supervision True --train_batch_size 16 --test_batch_size 16 --mode TXT >log/IRSTD-1k_1.log 2>&1 &

#### On Ubuntu:

#### 1. Train.

```bash
python train.py --base_size 256 --crop_size 256 --epochs 1500 --dataset NUDT-SIRST --split_method 50_50 --model DNANet --backbone resnet_18  --deep_supervision True --train_batch_size 16 --test_batch_size 16 --mode TXT

```
#### 2. Test.

```bash
python test.py --base_size 256 --crop_size 256 --st_model NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS --model_dir NUAA-SIRST_DNANet_28_07_2021_05_21_33_wDS/mIoU__DNANet_NUAA-SIRST_epoch.pth.tar --dataset NUAA-SIRST --split_method 50_50 --model DNANet --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optional 1) Visulize your predicts.
```bash
python visulization.py --base_size 256 --crop_size 256 --st_model [trained model path] --model_dir [model_dir] --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optional 2) Test and visulization.
```bash
python test_and_visulization.py --base_size 256 --crop_size 256 --st_model NUDT-SIRST_DNANet_04_07_2024_18_16_44_wDS --model_dir NUDT-SIRST_DNANet_04_07_2024_18_16_44_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar --dataset NUDT-SIRST --split_method 50_50 --model DNANet --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```

#### (Optional 3) Demo (with your own IR image).
```bash
python demo.py --base_size 256 --crop_size 256 --img_demo_dir [img_demo_dir] --img_demo_index [image_name]  --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT  --suffix [img_suffix]

```







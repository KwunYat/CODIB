# CODIB: Learning Compact Representations with an Information Bottleneck for Camouflaged Object Detection (TMM 2025)


> **Authors:** 
> Guanyi Li,
> Junjie Zhang,
> Rui Gao,
> Wubang Yuan,
> Gloria Jin, 
> and Dan Zeng.

## 1. Preface

- This repository provides code for "_**Learning Compact Representations with an Information Bottleneck for Camouflaged Object Detection**_" TMM-2025.

## 2. Proposed Baseline

### 2.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA GeForce RTX 3090Ti GPU.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n CODIB python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/CODIB/CODIB.pth`, 
    which can be found in this [download link (Baidu Drive)](https://pan.baidu.com/s/1tVno4D_aQnKKTRpuoMiHwQ?pwd=fzfx).
    
    + downloading PVTv2 weights and move it into `./models/pvt_v2_b2.pth`[download link (Baidu Drive)](https://pan.baidu.com/s/1aoabxPFaR2h4BMZSoXESuQ?pwd=2zs4).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `train.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 2.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

If you want to speed up the evaluation on GPU, you just need to use the [efficient tool](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.

Assigning your costumed path, like `method`, `mask_root` and `pred_root` in `eval.py`.

Just run `eval.py` to evaluate the trained model.

> pre-computed maps of CODIB can be found in [download link (Baidu Drive)](https://pan.baidu.com/s/128rakZjOai1JmH1fgE5eiQ?pwd=siyg).


## 3. Citation

Please cite our paper if you find the work useful: 



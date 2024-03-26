# CaBins
CaBins: CLIP-based Adaptive Bins for Monocular Depth Estimation
[paper link]

## Installation
```bash
conda create -n CaBins python=3.9 -y
conda activate CaBins 
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

pip install wandb scipy ftfy regex tqdm
```

## Datasets
We use [NYU-Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) datasets.
You can prepare datasets by following [BTS](https://github.com/cleinc/bts/tree/master) and modify the path in the argument file.

## Training
Training on NYU-Depth V2 dataset:
```bash
python train.py arguments_train_nyu.txt
```
Training on KITTI dataset:
```bash
python train.py arguments_train_eigen.txt
```

## Testing and Evaluation
Testing and Evaluation on NYU-Depth V2 dataset:
```bash
python evaluate.py arguments_test_nyu.txt
```
Testing and Evaluation on KITTI dataset:
```bash
python evaluate.py arguments_test_eigen.txt
```

## Pre-trained model
You can download pre-trained models on [NYU-Depth V2](https://drive.google.com/file/d/1zdx8H1YCt71D9zLpfiovvt08dHmBD-bJ/view?usp=sharing) and [KITTI](https://drive.google.com/file/d/1ZwW3I5qN6gqrxfoXrLf-18lvM8xkcymJ/view?usp=sharing) datasets for inference.

## Acknowledgements
Our code is built upon BTS and AdaBins. We're grateful for their publicly available works.
Additionally, we extend our thanks to CLIP, DenseCLIP, and DepthCLIP for their excellent contributions.

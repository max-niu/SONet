# SONet

A lightweight and fast RGB-Thermal crowd counting model designed for smart city edge computing. SONet uses a single-stream network with one-step fusion, achieving high accuracy with only ~17M parameters and significantly accelerated inference.

## Quick Start

### 1. Data Preparation

* Download the required datasets from the following public repositories: [DroneRGBT](https://github.com/VisDrone/DroneRGBT), [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting).

* Extract the downloaded dataset files.

* Run the following command for data preprocessing:

```
python preprocess.py [dataset_name] [dataset_path]
```

### 2. Model Training

* Start training the model:

```
python train.py --dataset_path [dataset_path] --save_path [save_path]
```

Training parameters can be configured in `args.py`. Logs and model weights will be saved under the specified [save_path].

### 3. Model Inference

* Run inference with a trained model:

```
python test.py --dataset_path [dataset_path] --save_path [results_path] --pretrain [path_to_weights]
```

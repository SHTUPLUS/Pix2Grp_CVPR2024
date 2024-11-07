

# Official Implementation of "From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models"

    
## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model Zoo](#modelzoo)
  - [Training](#train)
  - [Evaluation](#eval)
  - [Reference](#ref)

## Introduction

Our paper ["From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models"](https://arxiv.org/abs/2404.00906) has been accepted by CVPR 2024.


## Installation

1. Creating conda environment and install pytorch

```bash
conda create -n pix2sgg python=3.8
conda activate pix2sgg

# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# or CUDA 10.2
conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=10.2 -c pytorch
```

2. Install other dependencies:
```bash
pip install -r requirements_pix2sgg.txt
# the hugging face version: v4.29.2
```
Our work is built upon LAVIS, sharing the majority of its requirements.

3. Build Project
```bash
python setup.py build develop
```


## Datasets
Check [DATASET.md](DATASET.MD)  for instructions of dataset preprocessing.

## Model Zoo

### Open Vocabulary SGG
The model weight can be download from: https://huggingface.co/rj979797/PGSG-CVPR2024/tree/main

|              | **Novel+base** |             |   **Novel**  | **checkpoint** |
|--------------|:--------------:|:-----------:|:------------:|----------------|
| **Datasets** |  **mR50/100**  | **R50/100** | **mR50/100** |                |
| VG           |  6.2/8.3       |  15.1/18.4  |  3.7/5.2     |  vg_ov_sgg.pth |
| VG-SGCls     |  9.7/13.8      |  26.8/33.2  |  5.1/7.7     |  vg_ov_sgg.pth |
| PSG          | 15.3/17.7      | 23.7/25.4   |  6.7/9.6     |  psg_ov_sgg.pth|
<!-- | OIv6         |                |             |              |                | -->

### Close Vocabulary SGG

| **Datasets** | **mR50/100** | **R50/100** | **checkpoint** |
|--------------|:------------:|:-----------:|----------------|
| VG           |  9.0/11.5    | 17.7/ 20.7  |vg_sgg.pth   |
| PSG          |  14.5/17.6   | 25.8/28.9   | psg_sgg.pth   |
| VG-c         |  10.4/12.7   | 20.3/23.6   | vg_sgg_close_clser.pth  |
| PSG-c        |  21.2/22.0   | 34.9/36.1   | psg_sgg_close_clser.pth  |
<!-- | OIv6         |              |             |                | -->

## Training and  Evaluation

Our PGSG is trained using the BLIP pre-trained weights, accessible [here](
https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth). 

Ensure that the checkpoint path in the configuration file (*.yaml) is accurate before training or evaluation. During training, utilize the checkpoint specified by `model.pretrained`, while for evaluation, load the checkpoint specified by `model.finetuned`.


### VG dataset
#### Open Vocabulary SGG
Training 
```bash 
python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py  lavis/projects/blip/train/vrd_vg_ft_pgsg_ov.yaml --job-name VG-pgsg_ovsgg
```
Evaluation 
```bash
python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_vg_pgsg_eval_ov.yaml --job-name VG-pgsg_stdsgg-eval 
```

#### Standard SGG

Training 
```bash 
python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py  lavis/projects/blip/train/vrd_vg_ft_pgsg.yaml --job-name VG-pgsg_ovsgg
```

Evaluation 
```bash
python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_vg_pgsg_eval.yaml --job-name VG-pgsg_stdsgg-eval 
```



### PSG dataset

#### Open Vocabulary SGG
Training
```bash 
python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/vrd_psg_ft_pgsg_ov.yaml --job-name psg-pgsg_ovsgg
```
Evaluation
```bash
python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_psg_ov.yaml --job-name psg-pgsg_ovsgg-eval 
```
#### Standard SGG
Training
```bash 
python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/vrd_psg_ft_pgsg.yaml --job-name psg-pgsg_stdsgg
```
Evaluation
```bash
python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_psg_eval.yaml --job-name psg-pgsg_stdsgg-eval 
```

## Paper and Citing 
If you find this project helps your research, please kindly consider citing our papers in your publications. 

```bibtex
@misc{li2024pixels,
    title={From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models},
    author={Rongjie Li and Songyang Zhang and Dahua Lin and Kai Chen and Xuming He},
    year={2024},
    eprint={2404.00906},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledge

This repository is built on [LAVIS](https://github.com/salesforce/LAVIS) and borrows code from scene graph benchmarking framework from [SGTR](https://github.com/Scarecrow0/sgtr). 

## License
[BSD 3-Clause License](LICENSE.txt)

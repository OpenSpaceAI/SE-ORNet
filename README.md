## SE-ORNet: Self-Ensembling Orientation-aware Network for Unsupervised Point Cloud Shape Correspondence
PyTorch implementation for our CVPR 2023 paper SE-ORNet.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/se-ornet-self-ensembling-orientation-aware/3d-dense-shape-correspondence-on-shrec-19)](https://paperswithcode.com/sota/3d-dense-shape-correspondence-on-shrec-19?p=se-ornet-self-ensembling-orientation-aware)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

[[Project Webpage](https://chuxwa.github.io/SE-ORNet/)]
[[Paper](https://arxiv.org/abs/2304.05395)]

## News

* **28. February 2023**: SE-ORNet is accepted at CVPR 2023. :fire:
* **10. April 2023**: [SE-ORNet preprint](https://arxiv.org/abs/2304.05395) released on arXiv.
* **Coming Soon**: Code will be released soon.

## Running the code
To be supplemented

## Code structure

```
├── SE-ORNet
│   ├── __init__.py
│   ├── train.py     <- the main file
│   ├── models
│   │   ├── metrics       
│   │   ├── modules       
│   │   ├── runners    
│   │   ├── correspondence_utils.py  
│   │   ├── data_augment_utils.py
│   │   └── shape_corr_trainer.py   
│   ├── utils
│   │   ├── __init__.py      
│   │   ├── argparse_init.py   
│   │   ├── cyclic_scheduler.py   
│   │   ├── model_checkpoint_utils.py
│   │   ├── pytorch_lightning_utils.py
│   │   ├── switch_functions.py
│   │   ├── tensor_utils.py
│   │   └── warmup.py
│   ├── visualization
│   │   ├── __init__.py
│   │   ├── mesh_container.py
│   │   ├── mesh_visualization_utils.py
│   │   ├── mesh_visualizer.py
│   │   ├── orca_xvfb.bash
│   │   └── visualize_api.py    
│   └── ChamferDistancePytorch
├── data
│   ├── point_cloud_db
│   ├── __init__.py
│   └── generate_smal.md
├── .gitignore
├── .gitmodules
├── README.md
└── LICENSE
```

## Dependencies
The main dependencies of the project are the following:
```yaml
python: 3.7
cuda: 11.1
```
## Datasets
The method was evaluated on:
* SURREAL
  * 230k shapes (DPC uses the first 2k).
  * [Dataset website](https://www.di.ens.fr/willow/research/surreal/data/)
  * This code downloads and preprocesses SURREAL automatically.

* SHREC’19
  * 44 Human scans.
  * [Dataset website](http://3dor2019.ge.imati.cnr.it/shrec-2019/)
  * This code downloads and preprocesses SURREAL automatically.

* SMAL
  * 10000 animal models (2000 models per animal, 5 animals).
  * [Dataset website](https://smal.is.tue.mpg.de/)
  * Due to licencing concerns, you should register to [SMAL](https://smal.is.tue.mpg.de/) and download the dataset.
  * You should follow data/generate_smal.md after downloading the dataset.
  * To ease the usage of this benchmark, the processed dataset can be downloaded from [here](https://mailtauacil-my.sharepoint.com/:f:/g/personal/dvirginzburg_mail_tau_ac_il/Ekm37j0fi71Fn305v9nmXHABCSc1mWFa17uAc2jOngcyTQ?e=Ns2InB). Please extract and put under `data/datasets/smal`

* TOSCA
  * 41 Animal figures.
  * [Dataset website](http://tosca.cs.technion.ac.il/book/resources_data.html)
  * This code downloads and preprocesses TOSCA automatically.
  * To ease the usage of this benchmark, the processed dataset can be downloaded from [here](https://mailtauacil-my.sharepoint.com/:f:/g/personal/dvirginzburg_mail_tau_ac_il/EoMgplq-XqlGpl6K6lW6C8gBCxfq2gWXQ4f94xchF3dc9g?e=USid0X). Please extract and put under `data/datasets/tosca`

## Data preprocessing
To be supplemented

## Training & inference

For training run
``` 
python train_point_corr.py --dataset_name <surreal/tosca/shrec/smal>
```
The code is based on [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), all PL [hyperparameters](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html) are supported. 

For testing, simply add `--do_train false` flag, followed by `--resume_from_checkpoint` with the relevant checkpoint.

```
python train_point_corr.py --do_train false  --resume_from_checkpoint <path>
```
Test phase visualizes each sample, for faster inference pass `--show_vis false`.

We provide a trained checkpoint repreducing the results provided in the paper, to test and visualize the model run
``` 
python train_point_corr.py --show_vis --do_train false --resume_from_checkpoint data/ckpts/surreal_ckpt.ckpt
```

## BibTeX
To be supplemented

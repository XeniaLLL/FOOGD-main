# FOOGD
This repository is an official PyTorch implementation of paper:
[FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Liao_Rethinking_the_Representation_in_Federated_Unsupervised_Learning_with_Non-IID_Data_CVPR_2024_paper.html).
NeurIPS 2024 (Poster).

[//]: # (Thanks to [@Pengyang]&#40;https://github.com/PengyangZhou&#41; and [@Fengyuan]&#40;https://github.com/anonymifish&#41; for providing a robust and practical implementation framework.)

# FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection

## Abstract
Federated learning (FL) is a promising machine learning paradigm that collaborates
with client models to capture global knowledge. However, deploying FL models in
real-world scenarios remains unreliable due to the coexistence of in-distribution
data and unexpected out-of-distribution (OOD) data, such as covariate-shift and
semantic-shift data. Current FL researches typically address either covariate-shift
data through OOD generalization or semantic-shift data via OOD detection, overlooking the simultaneous occurrence of various OOD shifts. In this work, we
propose FOOGD, a method that estimates the probability density of each client
and obtains reliable global distribution as guidance for the subsequent FL process. Firstly, SM3D in FOOGD estimates score model for arbitrary distributions
without prior constraints, and detects semantic-shift data powerfully. Then SAG in
FOOGD provides invariant yet diverse knowledge for both local covariate-shift
generalization and client performance generalization. In empirical validations,
FOOGD significantly enjoys three main advantages: (1) reliably estimating non-normalized decentralized distributions, (2) detecting semantic shift data via score
values, and (3) generalizing to covariate-shift data by regularizing feature extractor.
## Implementation 
This project backbone is contributively implemented by my co-authors, [@Pengyang](https://github.com/PengyangZhou), [@Fengyuan](https://github.com/anonymifish) and [@Jiahe](https://github.com/Che-Xu).


[//]: # (### Step 1: install FL toolbox implemented by my co-author Huabin)

[//]: # (```)

[//]: # (git clone https://github.com/zhb2000/fedbox.git)

[//]: # (cd fedbox)

[//]: # (pip install .)

[//]: # (```)

[//]: # (### Step 2: train model)

[//]: # (Assign parameters for `train_xxx.py` that is called in `src/main.py`)

[//]: # (```)

[//]: # (# execute FedU2)

[//]: # (python src/main.py )

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (## Citation)

[//]: # (If you find HyperFed useful or inspiring, please consider citing our paper:)

[//]: # (```bibtex)

[//]: # (@InProceedings{Liao_2024_CVPR,)

[//]: # (    author    = {Liao, Xinting and Liu, Weiming and Chen, Chaochao and Zhou, Pengyang and Yu, Fengyuan and Zhu, Huabin and Yao, Binhui and Wang, Tao and Zheng, Xiaolin and Tan, Yanchao},)

[//]: # (    title     = {Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data},)

[//]: # (    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[//]: # (    month     = {June},)

[//]: # (    year      = {2024},)

[//]: # (    pages     = {22841-22850})

[//]: # (})

[//]: # (```)




## dataset

Here are links for the OOD dataset used in project:
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html),
[LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz),
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

You can download corrupted CIFAR-10 dataset `CIFAR-10-C` via this link
[CIFAR-10-C](https://zenodo.org/records/2535967).

Download and unzip required datasets in your own folder `/path/to/dataset`
(don't change folder's name if you don't want to adapt codes)

**Noted: CIFAR-10 and CIFAR-100 can be downloaded via `torchvision`. You don't
have to download them by yourself if you want to use them. Specifying where to save them
is all you need.**
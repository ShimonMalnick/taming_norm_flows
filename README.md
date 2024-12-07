<h1 style="text-align: center;">
Taming Normalizing Flows <br>
WACV 2024
</h1>

<a href="https://www.malnick.net/taming_norm_flows/"><img src="https://img.shields.io/badge/Project-Website-green"></a>
<a href="https://arxiv.org/abs/2211.16488"><img src="https://img.shields.io/badge/arXiv-2211.16488-b31b1b.svg"></a>
<a href="https://openaccess.thecvf.com/content/WACV2024/html/Malnick_Taming_Normalizing_Flows_WACV_2024_paper.html"><img src="https://img.shields.io/badge/WACV-2024-blue"></a>

<img src="images/toy_example.jpg">

<h3 style="text-align: center;">
<a href="https://www.malnick.net/">Shimon Malnick</a>,
<a href="http://www.eng.tau.ac.il/~avidan/"> Shai Avidan</a> and
<a href="https://www.ohadf.com/"> Ohad Fried</a>
</h3>

> We propose an algorithm for taming Normalizing Flow models - changing the probability that the model will produce a specific image or image category. We focus on Normalizing Flows because they can calculate the exact generation probability likelihood for a given image. We demonstrate taming using models that generate human faces, a subdomain with many interesting privacy and bias considerations. Our method can be used in the context of privacy, e.g., removing a specific person from the output of a model, and also in the context of debiasing by forcing a model to output specific image categories according to a given distribution. Taming is achieved with a fast fine-tuning process without retraining from scratch, achieving the goal in a matter of minutes. We evaluate our method qualitatively and quantitatively, showing that the generation quality remains intact, while the desired changes are applied.


<div style="text-align: center;">
Official implementation of the paper:
 <a href="https://openaccess.thecvf.com/content/WACV2024/html/Malnick_Taming_Normalizing_Flows_WACV_2024_paper.html"> Taming Normalizing Flows</a>
</div>

# Setup
## 1. Download Data
- Download the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.<br>
- Download the base [Glow model].(https://drive.google.com/file/d/1lRyqHcqfTTTu_-zEa_Gi9aOmKsFlXD0y/view?usp=sharing)
- Download the pretrained [ArcFace face vlassifier].(https://drive.google.com/file/d/10R86EgslUUmeWWIn5gNjuG_Sh_Qa0ltB/view?usp=sharing)
- Download the [pretrained celebA attribute classifier](https://drive.google.com/file/d/1HrBnre1AW-UDlvr7G2z-YPuIDV5CGTnL/view?usp=sharing).
## 2. Create Environment
```shell
conda create -n taming python=3.11
conda activate taming
cd taming_norm_flows
pip install -r requirements.txt
```

## 3. Set variables in setup.py
```python
CELEBA_ROOT="/path/to/celeba/dataset"
ARCFACE_CKPT="/path/to/arcface/model"
BASE_MODEL_PATH="/path/to/base/pretrained/model"
```

# Run
## Taming an identity
the `forget_identity` attribute in the `configs/forget_identity.json` determines the id of the identity in celeba to forget. See all parameters by running `python taming_an_identity -h`
```shell
python taming_an_identity.py --config configs/forget_identity.json
```

## Taming an attribute
Will be uploaded soon
<!-- the `forget_attribute` attribute in the `configs/forget_attribute.json` determines which attribute in celeba to forget. See all parameters by running `python taming_an_attribute -h`
```shell
python taming_an_attribute.py --config configs/forget_attribute.json
``` -->

# Acknowledgments
This repository builds on code from the  on the [glow-pytorch](https://github.com/rosinality/glow-pytorch) repository. In addition, we also use code from [ArcFace](https://github.com/TreB1eN/InsightFace_Pytorch/tree/master) for a face classifer.

# BibTex
```bib
@inproceedings{malnick2024taming,
  title={Taming Normalizing Flows},
  author={Malnick, Shimon and Avidan, Shai and Fried, Ohad},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4644--4654},
  year={2024}
}
```
---
markdown:
  image_dir: /assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false #是否使用绝对（相对于项目文件夹）图片路径
---


# Ternarized  Neural Network for Image Classification

This repository contains a Pytorch implementation of the paper "[Optimize Deep Convolutional Neural Network with Ternarized Weights and High Accuracy](https://arxiv.org/abs/1807.07948)".

If you find this project useful to you, please cite [our work](https://arxiv.org/abs/1807.07948):
<!-- ```
Zhezhi He, Boqing Gong, and Deliang Fan. 
"Optimize Deep Convolutional Neural Network with Ternarized Weights and High Accuracy." 
IEEE Winter Conference on Applications of Computer Vision (WACV) 2019.
``` -->
```
@article{he2018optimize,
  title={Optimize Deep Convolutional Neural Network with Ternarized Weights and High Accuracy},
  author={He, Zhezhi and Gong, Boqing and Fan, Deliang},
  journal={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```

## Table of Contents
- [Dependencies](#Dependencies)
- [Usage](#Usage)
- [Results](#Results)
- [Methods](#Methods)


## Dependencies:

* Python 3.6 (Anaconda)
* Pytorch 4.1

The installation of pytorch environment could follow the steps in .... :+1:


$$ g \sim \mathcal{G}(0,1) $$

## Usage

```bash {.line-numbers}
python main.py --
```

## Results

　Some experimental results are shown here

## Methods

## Task list
- [x] Upload Trained models for CIFAR-10 and ImageNet datasets.


- [ ] Encoding the weights of residual expansion layers to further reduce the model size (i.e., memory usage).

- [ ] Optimizing the thresholds chosen for the residual expansion layers.



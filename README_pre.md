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
  - ResNet-20/32/44/56 on CIFAR-10
  - AlexNet and ResNet-18/34/50/101 on ImageNet
- [Methods](#Methods)


## Dependencies:

* Python 3.6 (Anaconda)
* Pytorch 4.1


## Usage

For training the new model or evaluating the pretrained model, please use the following command in terminal. Remeber to revise the bash code with correct dataset/model path.

CIFAR-10:
```bash {.line-numbers}
bash train_CIFAR10.sh 
```

ImageNet:
```bash {.line-numbers}
bash train_ImageNet.sh
```

## Results
Trained models can be downloaded with the links provided (Google Drive).
### ResNet-20/32/44/56 on CIFAR-10:

|      | ResNet-20 | ResNet-32 | ResNet-44 | ResNet-56 |
|:----:|:---------:|:---------:|:---------:|:---------:|
|  Full-Precison  |           |           |           |           |
| Ternarized |           |           |           |           |

### AlexNet on ImageNet:
|                | First and Last Layer | Top1/Top5 Accuracy |
|:--------------:|:--------------------:|:------------------:|
|  [AlexNet (FP.)]()  |          Full-Precision          |    61.78%/82.87%   |
| [AlexNet (Tern.)]() |          Full-Precision          |    58.59%/80.44%   |
| [AlexNet (Tern.)]() |         Ternarized         |    57.15%/79.42%   |

### ResNet-18/34/50/101 on ImageNet:

|      | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 |
|:----:|:---------:|:---------:|:---------:|:----------:|
|  Full-Precision  |     69.75%/89.07%     |      73.31%/91.42%     |       76.13%/92.86%    |     77.37%/93.55%       |
| Ternarized |     66.01%/86.78%      |     70.95%/89.89%      |      74.00%/91.77%     |      75.63%/92.49%      |


**ResNet-18 on ImageNet with Residual Expansion Layer (REL):**
For reducing the accuracy drop caused by the aggresive model compression, we append the residual expansion layers to compensate the accuracy gap. Considering the aforementioned ternarized ResNet-18 is **t_ex=1** (i.e. without REL).

|   ResNet-18     | first and last layer | Top1/Top5 Accuracy |
|:------:|:--------------------:|:------------------:|
| [t_ex=2]() |         Tern         |    68.05%/88.04%   |
| [t_ex=4]() |         Tern         |    69.44%/88.91%   |

## Methods

## Task list
- [x] Upload Trained models for CIFAR-10 and ImageNet datasets.


- [ ] Encoding the weights of residual expansion layers to further reduce the model size (i.e., memory usage).

- [ ] Optimizing the thresholds chosen for the residual expansion layers.



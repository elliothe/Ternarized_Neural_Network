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
* [TensorboardX](https://github.com/lanpa/tensorboardX)


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
The entire network is ternarized (including first and last layer) for ResNet-20/32/44/56 on CIFAR-10. Note that, all the CIFAR-10 experiments are directly training from scratch, where no pretrained model is used. Users can ternarized the model from the pretrained model. Since CIFAR-10 is a toy dataset, I did not upload the trained model.

|      | ResNet-20 | ResNet-32 | ResNet-44 | ResNet-56 |
|:----:|:---------:|:---------:|:---------:|:---------:|
|  Full-Precison  |           |           |           |           |
| Ternarized |           |           |           |           |

### AlexNet on ImageNet:
|                | First and Last Layer | Top1/Top5 Accuracy |
|:--------------:|:--------------------:|:------------------:|
|  AlexNet (Full-Precision)  |          Full-Precision          |    [61.78%/82.87%](https://drive.google.com/open?id=1nu6FYzSDSw3YR6O2R2ZeNOZPTf8Xz0wX)   |
|  AlexNet (Ternarized) |          Full-Precision          |    [58.59%/80.44%](https://drive.google.com/open?id=1jfjtN-l9h8JxAgZ7l6e9ND5Y3fEfudMc)   |
| AlexNet (Ternarized) |         Ternarized         |    [57.21%/79.41%](https://drive.google.com/open?id=1XI9kw1AzMmOXGhpx93HXIRgwBIKJO3O9)   |

### ResNet-18/34/50/101 on ImageNet:

The pretrained models of full-precision baselines are from [Pytorch](https://github.com/pytorch/examples/tree/master/imagenet). 

|      | ResNet-18 | ResNet-34 | ResNet-50 | 
|:----:|:---------:|:---------:|:---------:|
|  Full-Precision  |     69.75%/89.07%     |      73.31%/91.42%     |       76.13%/92.86%    |    
| Ternarized |     [66.01%/86.78%](https://drive.google.com/open?id=1jCBfMDeSSHXBBXYYTfr1PAHRuOQpmpSh)      |     [70.95%/89.89%](https://drive.google.com/open?id=1dLQTy5jq5BIVbkWs1lEeqZKkyPBmmHvS)      |      [74.00%/91.77%](https://drive.google.com/open?id=1beceMaHgP0d8CzMqsbuttfynbXfxnU6R)     | 


**ResNet-18 on ImageNet with Residual Expansion Layer (REL):**
For reducing the accuracy drop caused by the aggresive model compression, we append the residual expansion layers to compensate the accuracy gap. Considering the aforementioned ternarized ResNet-18 is **t_ex=1** (i.e. without REL).

|   ResNet-18     | first and last layer | Top1/Top5 Accuracy |
|:------:|:--------------------:|:------------------:|
| t_ex=2 |         Tern         |    [68.35%/88.20%](https://drive.google.com/open?id=1u3ViNml0xCPO2kUcQSif4ttLMnZCExiT)   |
| t_ex=4 |         Tern         |    [69.44%/88.91%](https://drive.google.com/open?id=1jHqLQHmyU9B7oLlSmPi3Q06OJBkhS6iA)   |

<!-- ## Methods -->

<!-- ## Miscellaneous tools -->

## Task list
- [x] Upload Trained models for CIFAR-10 and ImageNet datasets.


- [ ] Encoding the weights of residual expansion layers to further reduce the model size (i.e., memory usage).

- [ ] Optimizing the thresholds chosen for the residual expansion layers.



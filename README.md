  
  
  
# Ternarized  Neural Network for Image Classification
  
  
This repository contains a Pytorch implementation of the paper "[Optimize Deep Convolutional Neural Network with Ternarized Weights and High Accuracy](https://arxiv.org/abs/1807.07948 )".
  
If you find this project useful to you, please cite [our work](https://arxiv.org/abs/1807.07948 ):
  
  
```
@article{he2018optimize,
  title={Optimize Deep Convolutional Neural Network with Ternarized Weights and High Accuracy},
  author={He, Zhezhi and Gong, Boqing and Fan, Deliang},
  journal={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```
  
## Table of Contents
  
- [Dependencies](#Dependencies )
- [Usage](#Usage )
- [Results](#Results )
- [Methods](#Methods )
  
  
## Dependencies:
  
  
* Python 3.6 (Anaconda)
* Pytorch 4.1
  
The installation of pytorch environment could follow the steps in .... :+1:
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?g%20&#x5C;sim%20&#x5C;mathcal{G}(0,1)"/></p>  
  
  
## Usage
  
  
```bash
python main.py --
```
  
## Results
  
  
　Some experimental results are shown here
  
## Methods
  
  
## Task list
  
- [x] Upload Trained models for CIFAR-10 and ImageNet datasets.
  
  
- [ ] Encoding the weights of residual expansion layers to further reduce the model size (i.e., memory usage).
  
- [ ] Optimizing the thresholds chosen for the residual expansion layers.
  
  
  
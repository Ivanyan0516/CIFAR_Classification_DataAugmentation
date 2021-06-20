# Data augmentation on CIFAR-10 image classification

This repository  is our final project on the deep learning and neural network course.

Our team members are Yanfan\Zhaobin\Liyingxue\Hongyuxin.

## Introduction

- **Cutout** is a simple regularization method for convolutional neural networks which consists of masking out random sections of input images during training. This technique simulates occluded examples and encourages the model to take more minor features into consideration when making decisions, rather than relying on the presence of a few major features.  

- **Mixup**  is a generic and straightforward data augmentation principle. In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples.

-  **CutMix**  patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. By making efficient use of training pixels and retaining the regularization effect of regional dropout, CutMix consistently outperforms the state-of-the-art augmentation strategies on CIFAR  classification tasks.

![image](https://github.com/Ivanyan0516/nndl-pj/tree/master/images/1.png)

![image](https://github.com/Ivanyan0516/nndl-pj/tree/master/images/2.png)

## Results and Usage

### Dependencies

[PyTorch v0.4.0](http://pytorch.org/)  
[tqdm](https://pypi.python.org/pypi/tqdm)

### Result 
| CIFAR-10 classification | train_acc | test_acc |
| ----------------------- | --------- | -------- |
| ResNet-18               | 1.0       | 0.8915   |
| ResNet-18 + cutout      | 0.9998    | 0.9020   |
| ResNet-18 + cutmix      | 0.8802    | 0.9073   |
| ResNet-18 + mixup       | 0.7476    | 0.9604   |

### Code

To train ResNet18 on CIFAR10 with  cutout:    
`python train.py --dataset cifar10 --model resnet18  --cutout --length 16`

To train ResNet18 on CIFAR10 with cutmix:  
`python train_cutmix.py --dataset cifar10 --model resnet18 --beta 1 --cutmix_prob 1`

To train ResNet18 on CIFAR10 with mixup:  

`python train_mixup.py --dataset cifar10 --model resnet18  --mixup `

To train ResNet18 on CIFAR10(baseline):

`python train.py --dataset cifar10 --model resnet18  `


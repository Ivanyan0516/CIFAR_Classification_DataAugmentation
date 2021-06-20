# Cutout

This repository contains the code for the paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552). 

这是我们小组的期末pj作业

## Introduction

**Cutout** is a simple regularization method for convolutional neural networks which consists of masking out random sections of input images during training. This technique simulates occluded examples and encourages the model to take more minor features into consideration when making decisions, rather than relying on the presence of a few major features.  



**Mixup**  is a generic and straightforward data augmentation principle. In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples.

 **CutMix**  patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. By making efficient use of training pixels and retaining the regularization effect of regional dropout, CutMix consistently outperforms the state-of-the-art augmentation strategies on CIFAR  classification tasks.

![Cutout applied to CIFAR-10](https://github.com/uoguelph-mlrg/Cutout/blob/master/images/cutout_on_cifar10.jpg "Cutout applied to CIFAR-10")

Bibtex:  
```
@article{devries2017cutout,  
  title={Improved Regularization of Convolutional Neural Networks with Cutout},  
  author={DeVries, Terrance and Taylor, Graham W},  
  journal={arXiv preprint arXiv:1708.04552},  
  year={2017}  
}
```

## Results and Usage   
### Dependencies  
[PyTorch v0.4.0](http://pytorch.org/)  
[tqdm](https://pypi.python.org/pypi/tqdm)

### ResNet18  
Test error (%, flip/translation augmentation, mean/std normalization, mean of 5 runs) 

| **Network** | **CIFAR-10** | **CIFAR-100** |
| ----------- | ------------ | ------------- |
| ResNet18    | 4.72         | 22.46         |
| ResNet18 + cutout | 3.99   | 21.96         |

To train ResNet18 on CIFAR10 with data augmentation and cutout:    
`python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16`

To train ResNet18 on CIFAR100 with data augmentation and cutout:  
`python train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8`



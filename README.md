# CBAM: Convolutional Block Attention Module for CIFAR10 with ResNet backbone
This repository aims at reproducing the results from "[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)". We use the module coinjointly with the ResNet CNN architecture. The module is tested on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset which is an image classification task with 10 different classes.

## CBAM module
The CBAM module takes as input a tensor of feature maps of shape Channel x Height x Width and apply two self-attention mechanisms consecutively. It is implemented in the src/models/models/cbam.py file.


The first attention mechanism is applied *channel-wise*, in that we want to select the channels (or features) that are the more relevant independently from any spatial considerations (ChannelAttention class).

The second attention mechanism is applied along the two *spatial dimensions*. We want to select the more relevant locations in the feature maps independently from the channels (SpatialAttention class).


This module is independant from the CNN architecture and can be used as is with other projects.


## ResNet

As the backbone, we use a Resnet implementation taken from [there](https://github.com/kuangliu/pytorch-cifar). The available networks are: ResNet18,Resnet34, Resnet50, ResNet101 and ResNet152.

The *CBAM* module can be used *two different ways*:

It can be put in every blocks in the ResNet architecture, after the convolution part and before the residual part.

It can also be put at the end of the ResNet network, just before the Linear predictor. In that case it is used only for the final feature maps.

Both are available here.

## Run the project

The parameters, set in the file src/parameters/training.json are the followings:

* *batch_size*: the batch size for training
* *lr*: the learning rate for training
* *momentum*: network is trained with SGD which requires momentum
* *batch_every*: how often to log batch results in terminal
* *n_epochs*: number of epochs for training
* *num_workers*: num of workers for the dataloader
* *model_name*: name used for saving losses charts and trained models
* *load_model*: trained model checkpoint path, if not the empty string then resume training from last trained epoch up to n_epochs
* *save_every*: how often to save the model in training (in epochs)
* *reduction_ratio*: reduction ratio for the channel attention bottleneck default to 16
* *kernel_cbam*: kernel for convolution in spatial attention must be an odd number
* *use_cbam_block*: if 1 put CBAM block in every ResNet Block
* *use_cbam_class*: if 1 put CBAM block before the classifier
* *resnet_depth*: Resnet type in [18,34,50,101,152]



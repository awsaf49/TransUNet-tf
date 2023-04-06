## TransUNet
> Tensorflow Implementation of [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)
<img src="https://production-media.paperswithcode.com/social-images/hfPJrzzvUuaeIMvb.png" width=800>

## Installation
```shell
pip install transunet
```

## Usage
```py
from transunet import TransUNet
model = TransUNet(image_size=224, pretrain=True)
```
## Notebook
Refer to this [Kaggle Notebook](https://www.kaggle.com/code/awsaf49/uwmgi-transunet-2-5d-train-tf) for use case of TransUnet. It is mention worthy that this notebok won Google OSSS Expert Award using TransUnet!

## References 
* [TransUNet](https://github.com/Beckschen/TransUNet)(Official)
* [TransUnet](https://github.com/kenza-bouzid/TransUnet)
* [vit-keras](https://github.com/faustomorales/vit-keras)
* [ResNetV2](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet_v2.py#L28-L56)


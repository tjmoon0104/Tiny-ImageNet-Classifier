# Tiny-ImageNet-Classifier
Tiny-ImageNet Classifier using Pytorch

## Tiny-ImageNet

| Properties                  |           |
| --------------------------- | --------- |
| Number of Classes           | 200       |
| Number of training Images   | 500       |
| Number of validation Images | 50        |
| Number of test Images       | 50        |
| Image Size                  | (64,64,3) |

[Tiny-ImageNet]: https://tiny-imagenet.herokuapp.com/	"ddd"



## Step.1 Create Baseline Classifier

We will use pretrained ResNet18 model as our baseline model. 

Since ResNet18 is trained with 224x224 images and output of 1000 classes, we would have to modify the architecture to fit 64x64 images and output of 200 classes.


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

[Tiny-ImageNet][Tiny-ImageNet]

## Step.1 Create Baseline Classifier

We will use ResNet18 model as our baseline model. 



![](https://www.researchgate.net/profile/Paolo_Napoletano/publication/322476121/figure/tbl1/AS:668726449946625@1536448218498/ResNet-18-Architecture.png)



Since ResNet18 is trained with 224x224 images and output of 1000 classes, we would have to modify the architecture to fit 64x64 images and output of 200 classes.



#### Model with no pretrained weight

```python
#Load Resnet18
model_ft = models.resnet18()
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
```

Following is the loss function and optimization used for baseline model

```python
#Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

Following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/baseline_no_pretrain.png?raw=true)

With baseline model and pretrained weight, we were able to achieve 25.9% validation accuracy.



#### Model with pretrained weight

```python
#Load Resnet18 with pretrained weights
model_ft = models.resnet18(pretrained=True)
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
```

Same loss function and optimization was used.

Following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/baseline.png?raw=true)

With baseline model and pretrained weight, we were able to achieve 56.9% validation accuracy.

Reference [Baseline][Baseline] for detail python code.



## Step.2 Preprocessing

Validation accuracy increased from 25.9% to 56.9% by using pretrained weight from ImageNet. The validity of pretrained weight was confirmed, even though the image size was 64x64. For next step, we would like to observe  the validity of pretrained weight when we train the model with 224x224 images. Images has to be preprocessed from 64x64 to 224x224. We used bicubic interpolation to improve the quality of low resolution image when expanding it to 224x224. 

```python
import cv2

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)
```

Following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/resnet18_224.png?raw=true)

With ResNet18 model and pretrained weight with 224x224 preprocessed images, we were able to achieve 73.1% validation accuracy.



## Step.3 Finetuning

We achieved a classifier model with validation accuracy of 73.1%. However if we evaluate 64x64 validation images with this model, validation accuracy drops to 15.3%. This is due to difference in input image size. 



















[Tiny-ImageNet]: https://tiny-imagenet.herokuapp.com/	"Link to Tiny-ImageNet"
[Baseline]: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_Baseline.ipynb	"Link to Baseline"






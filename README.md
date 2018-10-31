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

We will use a ResNet18 model as our baseline model. 



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

| Model    | pretrained weight | Dataset | Validation Accuracy |
| -------- | ----------------- | ------- | ------------------- |
| ResNet18 | None              | 64x64   | 25.9%               |

#### Model with pretrained weight

```python
#Load Resnet18 with pretrained weights
model_ft = models.resnet18(pretrained=True)
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
```

Same loss function and optimization were used.

The following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/baseline.png?raw=true)

| Model    | pretrained weight | Dataset | Validation Accuracy |
| -------- | ----------------- | ------- | ------------------- |
| ResNet18 | ImageNet          | 64x64   | 56.9%               |

Reference [Baseline][Baseline] for detail python code.



## Step.2 Preprocessing

Validation accuracy increased from 25.9% to 56.9% by using pretrained weight from ImageNet. The validity of pretrained weight was confirmed, even though the image size was 64x64. For the next step, we would like to observe the efficacy of pretrained weight when we train the model with 224x224 images. Images have to be preprocessed from 64x64 to 224x224. We used bicubic interpolation to improve the quality of a low-resolution image when expanding it to 224x224.

```python
import cv2

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)
```

The following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/resnet18_224.png?raw=true)

| Model    | pretrained weight | Dataset | Validation Accuracy |
| -------- | ----------------- | ------- | ------------------- |
| ResNet18 | ImageNet          | 224x224 | 73.1%               |

Reference [224][224] for detail python code.



## Step.3 Finetuning

We achieved a classifier model with validation accuracy of 73.1%. However, if we evaluate 64x64 validation images with this model, validation accuracy drops to 15.3%. This drop happens due to the difference in input image size. 



In order to use the 64x64 image, we have to retrain the model with 64x64 images. We used the weight from the previous (224x224 trained) model.

```python
#Load ResNet18
model_ft = models.resnet18()
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
#Load weights from 224x224 trained model
model_ft.load_state_dict(torch.load('./models/resnet18_224_w.pt'))
```

The following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/resnet18_224_64.png?raw=true)

| Model    | pretrained weight | Dataset | Validation Accuracy |
| -------- | ----------------- | ------- | ------------------- |
| ResNet18 | 224x224 model     | 64x64   | 54.5%               |

Validation accuracy of this model was not as high as expected. It is even lower than the model trained from ImageNet pretrained weight. 



If we compare the output size of each convolutional layer, we can observe output size of 64x64 input image is much smaller than 224x224 input image. 

| Layer Name   | Output Size (Input 224x224x3) | Output Size (Input 64x64x3) | ResNet-18                   |
| ------------ | ----------------------------- | --------------------------- | --------------------------- |
| conv1        | 112x112x64                    | 32x32x64                    | 7x7, 64, stride=2, pad=3    |
| max pool     | 56x56x64                      | 16x16x64                    | 3x3, stride=2, pad=1        |
| layer1       | 56x56x64                      | 16x16x64                    | [3x3, 64] x 2, stride = 1   |
| layer2       | 28x28x128                     | 8x8x128                     | [3x3, 128] x2, stride = 2   |
| layer3       | 14x14x256                     | 4x4x256                     | [3x3, 256] x2, stride = 2   |
| layer4       | 7x7x512                       | 2x2x512                     | [3x3, 512] x2, stride = 2   |
| average pool | 1x1x512                       | 1x1x512                     | Adaptive Average Pooling(1) |



First layer of ResNet18 has stride of 2 followed by maxpool layer with stride of 2. This reduces the information of the image in the early stage of CNN. 



For fine tuning, we decided to reduce the kernel size to 3x3, stride to 1, and padding to 1. Then remove max pool layer to keep the output size.



| Layer Name   | Output Size (Input 64x64x3) | ResNet-18 FineTune          |
| ------------ | --------------------------- | --------------------------- |
| conv1        | 64x64x64                    | (3x3, 64, stride=1, pad=1)* |
| max pool     | --------------              | (Removed)*                  |
| conv2        | 64x64x64                    | [3x3, 64] x 2, stride = 1   |
| conv3        | 32x32x128                   | [3x3, 128] x2, stride = 2   |
| conv4        | 16x16x256                   | [3x3, 256] x2, stride = 2   |
| conv5        | 8x8x512                     | [3x3, 512] x2, stride = 2   |
| average pool | 1x1x512                     | Adaptive Average Pooling(1) |



After fine tuning the layer, we train the model with 64x64 images.

```python
#Load Resnet18 with pretrained weights
model_ft = models.resnet18()
#Finetune Final few layers
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
model_ft.maxpool = nn.Sequential()
#Load pretrained weight from 224x224 trained model
pretrained_dict = torch.load('./models/resnet18_224_w.pt')
```

The following figure shows the training and validation results. 

![](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/img/restnet18_224_64_fine_tune.png?raw=true)

| Model             | pretrained weight | Dataset | Validation Accuracy |
| ----------------- | ----------------- | ------- | ------------------- |
| ResNet18-FineTune | 224x224 model     | 64x64   | 72.3%               |

Reference [FineTune][Fine] for detail python code.













[Tiny-ImageNet]: https://tiny-imagenet.herokuapp.com/	"Link to Tiny-ImageNet"
[Baseline]: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_Baseline.ipynb	"Link to Baseline"
[224]: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_224.ipynb	"Link to Baseline"
[Fine]: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_224_finetune.ipynb	"Link to Baseline"






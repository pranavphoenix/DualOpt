# DualOpt
Dual Optimizer Training 

A variant of the [SWATS training paradigm](https://arxiv.org/abs/1712.07628) which uses two optimizers for training. 

## Install

```bash
$ pip install dualopt
```

## Usage
### Image Classification

```python
import dualopt, torch
from dualopt import classification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 10

#define model
#define datasets and dataloaders

top1 = []   #top1 accuracy
top5 = []   #top5 accuracy
traintime = []
testtime = []
counter = 20  # number of epochs without any improvement in accuracy before we stop training for each optimizer

PATH = 'saved_model.pth' #path to save model 

classification(model, trainloader, testloader, device, PATH, top1, top5, traintime, testtime,  num_classes = num_classes, set_counter = counter)

print('Finished Training')
print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f} -Top 5 Accuracy : {max(top5):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
```
# Post-Training

Experiments show that we get good results when training using data augmentations such as Trivial Augment. We found that subsequent post-training without using any data augmentations can further improve the results. 

## Usage


```python
import dualopt, torch, torchvision
import torchvision.transforms as transforms
from dualopt import classification
from dualopt import post_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 10

#define model

#set batch size according to GPU 
batch_size = 512

# transforms
transform_train_1 = transforms.Compose(
        [ transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_train_2 = transforms.Compose(
        [ 
            transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform_test = transforms.Compose(
        [ transforms.ToTensor(),
     transforms.Normalize((0.4941, 0.4853, 0.4507), (0.2468, 0.2430, 0.2618))])

#Dataset
trainset_1 = torchvision.datasets.CIFAR10(root='/workspace/', train=True, download=True, transform=transform_train_1)
trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2) #trainloader with augmentations

trainset_2 = torchvision.datasets.CIFAR10(root='/workspace/', train=True, download=True, transform=transform_train_2)
trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2) #trainloader for post-training without augmentations

testset = torchvision.datasets.CIFAR10(root='/workspace/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)


top1 = []   #top1 accuracy
top5 = []   #top5 accuracy
traintime = []
testtime = []
counter = 20  # number of epochs without any improvement in accuracy before we stop training for each optimizer

PATH = 'saved_model.pth' #path to save model 

classification(model, trainloader_1, testloader, device, PATH, top1, top5, traintime, testtime,  num_classes = num_classes, set_counter = counter)
print('Finished Training')

model.load_state_dict(torch.load(PATH))

post_train(model, trainloader_2, testloader, device, PATH, top1, top5, traintime, testtime, num_classes = num_classes, set_counter = counter)
print('Finished Training')

print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f} -Top 5 Accuracy : {max(top5):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
```

#### Cite the following paper  
```
@misc{jeevan2022convolutional,
      title={Convolutional Xformers for Vision}, 
      author={Pranav Jeevan and Amit sethi},
      year={2022},
      eprint={2201.10271},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

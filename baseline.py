import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from time import time
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# download CIFAR10 dataset
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor() # ToTensor() transforms the data to tensor type and rescale [0,255] uint8 to [0,1] float
)


test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# ### Step 1: Prepare Data

batch_size = 128
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# check the first batch of the dataset
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape, X.dtype) # 
    print("Shape of y: ", y.shape, y.dtype)
    break

    
nsamples = 10
imgs = training_data.data[0:nsamples,:]
# imgs = np.expand_dims(imgs)
imgs = torch.tensor(imgs)
imgs = imgs/255
print(imgs.shape)
labels = training_data.targets[0:nsamples]
classes_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


fig=plt.figure(figsize=(20,5),facecolor='w')
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
    plt.imshow(imgs[i, :, :, :], cmap=plt.get_cmap('gray'))
    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# ### Step 2: Define CNN

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel = 3, outchannel = 16, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)

model = ResNet18().to(device)

print(model)


# ### Step 3: Define loss function and the optimizer

import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 5e-4) 


# ### Step 4: Train the neural nets

epochs=10

for i in range(epochs): # iterate over epochs
    tic = time()
    model.train()
    train_loss=0
    for j, (X, y) in enumerate(train_dataloader): # iterate over batches 
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print learning process every 100 batches
        if j % 100 == 0:
            loss, current = loss.item(), j * len(X)
            print(f"epoch {i} batch {j} loss: {loss/batch_size:>7f}")
    
    train_time = time() - tic
    
    # print test results after every epochs    
    with torch.no_grad():
        model.eval()
        test_loss=0
        hit=0
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            hit += (pred.argmax(1) == y).sum().item()
        print(f"epoch {i} training time: {train_time:>3f}s, train loss: {train_loss/len(train_dataloader.dataset):>7f} test loss: {test_loss/len(test_dataloader.dataset):>7f} accuracy: {hit/len(test_dataloader.dataset) :>7f}")    






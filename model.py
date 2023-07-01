import time
import torch
import unittest
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型并将其移动到GPU
pretrained_model = models.resnet18(pretrained=True).to(device)


# 打印预训练模型的摘要信息
torchsummary.summary(pretrained_model, (3, 32, 32))

# 超参数
batch_size = 128
num_epochs = 20
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4
num_classes = 10

# 数据预处理和增强
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 划分训练集和验证集
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

# 加载训练集、验证集和测试集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 自监督学习预训练模型定义
class CPCModel(nn.Module):
    def __init__(self, num_classes):
        super(CPCModel, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x

# 线性分类器定义
class LinearClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(512, num_classes)  # 输入维度改为512

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将特征展平成一维向量
        x = self.fc(x)
        return x

# 创建线性分类器并将其移动到GPU
linear_classifier = LinearClassifier(num_classes).to(device)

# 训练函数
def train(model, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    best_valid_loss = float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        model.eval()
        valid_loss = 0.0
        valid_total = 0
        valid_correct = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += targets.size(0)
                valid_correct += predicted.eq(targets).sum().item()

            valid_loss /= len(valid_loader)
            valid_accuracy = 100.0 * valid_correct / valid_total
            valid_losses.append(valid_loss)
            valid_accs.append(valid_accuracy)

            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pth')

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")

    return train_losses, valid_losses, train_accs, valid_accs

# 测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100.0 * correct / total
    return test_accuracy

# 单元测试类
class TestCIFAR10(unittest.TestCase):
    def test_performance(self):
        # 自监督学习预训练模型
        cpc_model = CPCModel(num_classes)
        train_losses, valid_losses, train_accs, valid_accs = train(cpc_model, train_loader, valid_loader)

        # 可视化训练过程
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(valid_losses, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(valid_accs, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 线性分类器训练
        linear_classifier = LinearClassifier(num_classes)
        train(linear_classifier, train_loader, valid_loader)

        # 在测试集上评估模型性能
        accuracy = test(linear_classifier, test_loader)

        self.assertGreaterEqual(accuracy, 70.0)  # 验证准确率是否大于等于70%

if __name__ == '__main__':
    unittest.main()

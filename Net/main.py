import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision as tv
import torchvision.transforms as transforms

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# 定义对数据的预处理
pre_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

# 训练集
trainset = tv.datasets.CIFAR10(
                    root='data', 
                    train=True, 
                    download=False,
                    transform=pre_transform)

trainloader = t.utils.data.DataLoader(
                    trainset, 
                    batch_size=4,
                    shuffle=True, 
                    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
                    'data',
                    train=False, 
                    download=False, 
                    transform=pre_transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4, 
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 网络定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x

if __name__ == '__main__':
    # 实例化一个网络模型
    net = Net().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练
    t.set_num_threads(8)
    for epoch in range(2):  
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()   
            
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' \
                    % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    # 测试
    correct = 0
    total = 0
    with t.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
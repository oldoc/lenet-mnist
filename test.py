'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from lenet import lenet
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

testset = torchvision.datasets.ImageFolder(root='./data/testing', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')

# Model
print('==> Building model..')
net = torch.load('mnist.pth', map_location='cpu')
#net = net.module
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def test():

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            i = 0
            if True:
                for m in net._modules.values():
                    for j in m:
                        i = i + 1
                        if ((type(j) == nn.Conv2d) or (type(j) == nn.Linear)) and i == 11:
                            print(i)
                            w5 = j.weight.cpu().detach().numpy()
                            x4 = np.loadtxt("result.csv", delimiter=",")
                            #print(x4.dot(w5[7]))
                            #print(x4)

                            for i in range(x4.size):
                                x4[i] = x4[i] * w5[7][i]
                            with open('results.csv', 'a+') as f:
                                f.write(testset.imgs[batch_idx][0].split('/')[-1].split('.')[0]+',')
                                f.write(",".join(str(i) for i in x4) + '\n')
                                f.close()

                            # np.savetxt("weight.csv", w5, delimiter=",")
                            '''
                            with open('weight.csv', 'a+') as f:
                                ll = j.weight.reshape(j.weight.size(0), -1).cpu().detach().numpy().tolist()
                                f.write(",".join(str(i) for i in ll)+'\n')
                                f.close()
                            '''

    acc = 100.*correct/total
    print('Acc: %.2f' %(acc))



test()

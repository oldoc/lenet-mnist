'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
 
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=False),
            #nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, bias=False),
            #nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(120, 84, bias=False),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84, 10, bias=False),
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        out = x3.reshape(x3.size(0), -1)
        x4 = self.fc1(out)
        x5 = self.fc2(x4)

        if False:
            with open('result.csv', 'w') as f:
                f.write("，".join(str(i) for i in x1.reshape(x1.size(0), -1).cpu().numpy().tolist())+'\n')
                f.write("，".join(str(i) for i in x2.reshape(x2.size(0), -1).cpu().numpy().tolist())+'\n')
                f.write("，".join(str(i) for i in x3.reshape(x3.size(0), -1).cpu().numpy().tolist())+'\n')
                f.write("，".join(str(i) for i in x4.reshape(x4.size(0), -1).cpu().numpy().tolist())+'\n')
                f.write("，".join(str(i) for i in x5.reshape(x5.size(0), -1).cpu().numpy().tolist())+'\n')
                f.close()
        return x5


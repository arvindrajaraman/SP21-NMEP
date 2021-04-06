import torch
import torch.nn as nn
from torchsummary import summary

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        # populate the layers with your custom functions or pytorch
        # functions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResBlock(input_channels=64, output_channels=64, stride=1)
        self.layer2 = ResBlock(input_channels=64, output_channels=128, stride=2)
        self.layer3 = ResBlock(input_channels=128, output_channels=256, stride=2)
        self.layer4 = ResBlock(input_channels=256, output_channels=512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1)
        self.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.maxpool(out)
        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

    def new_block(self, in_planes, out_planes, stride):
        layers = [
            nn.Conv2d(in_channels=in_planes,  out_channels=out_planes, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1),

        ]
        #TODO: make a convolution with the above params
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=input_channels,  out_channels=output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        self.conv3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=output_channels)
        self.conv4 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=output_channels)
        if input_channels == output_channels:
            self.identity = lambda x : x 
        else:
            self.identity = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, stride=2, kernel_size=1, padding=0)
    
    def forward(self, x):
        # print('0', x.shape)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print('1', out.shape)
        # print('2', self.identity(identity).shape)
        out += self.identity(identity)
        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out += identity 
        out = self.bn4(out)
        out = self.relu(out)
        return out

# model = ResNet(None, None)
# summary(model, input_size=(3, 32, 32), batch_size=1)

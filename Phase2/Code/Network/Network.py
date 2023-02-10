"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = torch.nn.CrossEntropyLoss()(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      self.network = nn.Sequential(

          nn.Conv2d(3, 16, 3, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),  # Output (16 x 16 x 32)
          nn.Conv2d(32, 64, 3, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 128, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          nn.Flatten(),
          nn.Linear(8 * 8 * 128, 100),
          nn.ReLU(),
          nn.Linear(100, 10)
      )

      # updated Base model

      #             nn.Conv2d(3, 32, 3, padding=1),
      #             nn.ReLU(),
      #             nn.Dropout(0.5),
      #             nn.Conv2d(32, 64, 3, padding=1),
      #             nn.ReLU(),
      #             nn.Dropout(0.5),
      #             nn.MaxPool2d(2,2),
      #             nn.Conv2d(64, 128, 3, padding=1),
      #             nn.ReLU(),
      #             nn.Dropout(0.5),
      #             nn.Conv2d(128, 256, 3, padding=1),
      #             nn.ReLU(),
      #             nn.Dropout(0.5),
      #             nn.MaxPool2d(2,2),
      #             nn.Flatten(),
      #             nn.Linear(8*8*256,100),
      #             nn.ReLU(),
      #             nn.Linear(100,10)
      #             )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################
      out = self.network(xb)
      print(xb.shape)
      return out

  def training_step(self, batch):
      X, y = batch

      # X = X.to(device)
      # y = y.to(device)
      # Predict output with forward pass
      y_hat = self.forward(X)
      # get the loss
      loss_wrapper = loss_fn()
      loss = loss_wrapper(y_hat, y)
      return loss
      # return super().training_step(batch)

  def validation_step(self, batch):
      X, y = batch

      # X = X.to(device)
      # y = y.to(device)
      # Predict output with forward pass
      y_hat = self.forward(X)
      # get the loss
      loss_wrapper = loss_fn()
      loss = loss_wrapper(y_hat, y)
      acc = accuracy(y_hat, y)
      return {'loss': loss.detach(), 'acc': acc}
      # return loss
      # return super().training_step(batch)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ImageClassificationBase):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


ResNet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)

class ResNeXtBottleneck(ImageClassificationBase):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=10, cardinality=32):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.cardinality, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

ResNext = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], num_classes=10, cardinality=32)


class DenseNet(ImageClassificationBase):
    def __init__(self, in_channels, growth_rate, num_layers, num_classes):
        super(DenseNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        in_channels = growth_rate
        for i in range(len(num_layers)):
            self.dense_blocks.append(DenseBlock(in_channels, growth_rate, num_layers[i]))
            in_channels += growth_rate * num_layers[i]
            if i != len(num_layers) - 1:
                out_channels = int(in_channels * 0.5)
                self.transition_layers.append(Transition(in_channels, out_channels))
                in_channels = out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


# Create DenseNet model
DenseNet = DenseNet(in_channels=3, growth_rate=32, num_layers=[6, 12, 24, 16], num_classes=10)

import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, channels = 64, samples = 801, classes = 2):
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution (Acts like the Bandpass Filter)
        self.conv1 = nn.Conv2d(1,8, (1,64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)

        # Block 2: Depthwise Spatial Convolution (Acts like CSP)
        self.depthwise = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)

        # Block 3: Separable Convolution (Feature extraction)
        self.separable = nn.Conv2d(16, 16, (1, 16), padding='same', groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)

        #Classifier Setup
        self.flatten = nn.Flatten()

        # Dynamically calculate the flattened dimension size for the linear layer
        dummy_input = torch.randn(1, 1, channels, samples)
        with torch.no_grad():
            x_dummy = self.pool2(self.elu2(self.batchnorm3(self.pointwise(self.separable(
                      self.dropout1(self.pool1(self.elu1(self.batchnorm2(self.depthwise(
                      self.batchnorm1(self.conv1(dummy_input))))))))))))
            flattened_size = x_dummy.shape[1] * x_dummy.shape[2] * x_dummy.shape[3]
            
        self.classifier = nn.Linear(flattened_size, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.separable(x)
        x = self.pointwise(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    

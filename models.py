## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Input layer - dimension is 1 x 224 x 224 (C x H x W)
        
        # Hidden layers
        # 1st layer (conv+relu) - 32 convolution filters of size 5x5 (each of depth 1 and stride 1) - output dimension = 32 x 220 x 220
        # 2nd layer (pool) - 32 max-pooling filters of size 2x2 (each of depth 1 and stride 2) - output dimension = 32 x 110 x 110
        # 3rd layer (conv+relu) - 64 convolution filters of size 5x5 (each of depth 32 and stride 1) - output dimension = 64 x 106 x 106
        # 4th layer (pool) - 64 max-pooling filters of size 2x2 (each of depth 1 and stride 2) - output dimension = 64 x 53 x 53
        # Dropout 2d with probability 0.3
        # 5th layer (conv+relu) - 128 convolution filters of size 3x3 (each of depth 64 and stride 1) - output dimension = 128 x 51 x 51
        # 6th layer (pool) - 128 max-pooling filters of size 2x2 (each of depth 1 and stride 2) - output dimension = 128 x 25 x 25
        # 7th layer (conv+relu) - 256 convolution filters of size 3x3 (each of depth 128 and stride 1) - output dimension = 256 x 23 x 23
        # 8th layer (pool) - 256 max-pooling filters of size 2x2 (each of depth 1 and stride 2) - output dimension = 256 x 11 x 11
        # Dropout 2d with probability 0.4        
        # 9th layer (fc+relu)- Fully-connected with 1000 nodes, with 256*11*11 input dimension
        # Dropout with probability 0.5    
        # 10th layer (fc+relu) - Fully-connected with 1000 nodes, with 1000 input dimension
        # Dropout with probability 0.6
        
        # Output layer - Fully-connected with 136 nodes, with 1000 input dimension
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # output dimension = 32 x 220 x 220
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5) # output dimension = 64 x 106 x 106
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3) # output dimension = 128 x 51 x 51        
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3) # output dimension = 256 x 23 x 23
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(in_features = 256*11*11, out_features = 1000) # input dim is of the flattened output of max-pooling on conv4
        self.fc2 = nn.Linear(in_features = 1000, out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000, out_features = 136)
                
        # Dropouts
        self.drop_conv2 = nn.Dropout2d(p=0.3)
        self.drop_conv4 = nn.Dropout2d(p=0.4)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.drop_fc2 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        
        # The convolutional layers with relu/pooling/dropouts
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop_conv2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop_conv4(x)
        
        # Flatten before feeding into the dense layer
        x = x.view(x.size(0), -1) 
        
        # The dense layers
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_fc2(x)        
        x = self.fc3(x)
        
        
        # Return the modified x, having gone through all the layers of the model
        return x

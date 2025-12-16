import torch
import torch.nn as nn

# represents one convolutional block:
# Conv -> BatchNorm -> ReLU
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class SmallCIFARCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        
        self.features = nn.Sequential(
            #Block 1: 32x32 -> 16x16
            ConvBNReLU(3,64),
            ConvBNReLU(64,64),
            nn.MaxPool2d(2), #32->16
            
            # Block 2: 16x16->8x8
            ConvBNReLU(64,128),
            ConvBNReLU(128,128),
            nn.MaxPool2d(2), #16->8
            
            # Block 3
            ConvBNReLU(128,256),
            ConvBNReLU(256,256)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(256,num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
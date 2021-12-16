import torch
import torch.nn as nn
import torch.nn.functional as F



class depthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution module, in `modified` Xception configuration.
    """
    def __init__(self, n_in, n_out, kernel_size, padding, bias=False, mode: str="modified"):
        """
        - mode: 'original' or 'modified' depthwise separable convolution.
        """
        super(depthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, kernel_size=kernel_size, padding=padding, groups=n_in, bias=bias)
        self.pointwise = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias)
        self.mode = mode

    def forward(self, x):
        if self.mode == "modified":
            out = self.pointwise(x)
            out = self.depthwise(out)
        else:
            # original depthwise spearable convolution, performs 
            # the depthwise convolution before the pointwise convolution
            out = self.depthwise(x)
            out = self.pointwise(out)
        return out


class EntryFlow(nn.Module):
    """
    Xception Entry Flow module.
    """
    def __init__(self, input_channel):
        super(EntryFlow, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block_2 = nn.Sequential(
            depthwiseSeparableConv(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            depthwiseSeparableConv(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        
        self.block_3 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            
            nn.ReLU(True),
            depthwiseSeparableConv(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        
        self.block_4 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(256, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwiseSeparableConv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        out1 = self.block_1(x)
        out2 = self.block_2(out1) + self.block_2_residual(out1)
        out3 = self.block_3(out2) + self.block_3_residual(out2)
        out = self.block_4(out3) + self.block_4_residual(out3)
        return out


class MiddleFlow(nn.Module):
    """
    Xception Middle Flow module.
    """
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.activation     = nn.ReLU(True)
        self.separable_conv = depthwiseSeparableConv(728, 728, 3, 1)
        self.normalization  = nn.BatchNorm2d(728)
    
    def forward(self, x):
        # first block
        out1 = self.activation(x)
        out1 = self.separable_conv(out1)
        out1 = self.normalization(out1)

        # first block
        out2 = self.activation(out1)
        out2 = self.separable_conv(out2)
        out2 = self.normalization(out2)

        # first block
        out3 = self.activation(out2)
        out3 = self.separable_conv(out3)
        out = self.normalization(out3) + x

        return out


class ExitFlow(nn.Module):
    """
    Xception Exit Flow module without the fully connected layers at the end.
    """
    def __init__(self, num_classes: int=10):
        super(ExitFlow, self).__init__()
        self.block_1 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwiseSeparableConv(728, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.block_2 = nn.Sequential(
            depthwiseSeparableConv(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),
            
            depthwiseSeparableConv(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        out1 = self.block_1(x) + self.block_1_residual(x)
        out2 = self.block_2(out1)

        avg_pool = F.adaptive_avg_pool2d(out2, (1, 1))                
        # flatten the output
        output = avg_pool.view(avg_pool.size(0), -1)

        return output


class Xception(nn.Module):
    """
    Xception architecture pytorch module.
    """
    def __init__(self, in_channel, num_classes: int=10, middle_rep: int=8):
        super(Xception, self).__init__()
        self.entry_flow  = EntryFlow(input_channel=in_channel)
        self.middle_flow = MiddleFlow()
        self.exit_flow   = ExitFlow(num_classes=num_classes)

        self.fc = nn.Linear(2048, num_classes)
        self.middle_rep  = middle_rep 

    @staticmethod
    def get_classifiers():
        # arch-depth-widenFactor
        return ['xc-8-4', 'xc-4-4', 'xc-12-4']
    
    @classmethod
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):
        _, depth, narrowing = arch.split('-')
        cls_instance = cls(input_channels=input_channels, num_classes=num_classes, middle_rep=int(depth))
        return cls_instance

    def forward(self, x):
        # entry flow
        entry_out = self.entry_flow(x) 

        # middle flow
        middle_out = self.middle_flow(entry_out)
        for i in range(self.middle_rep-1):
            middle_out = self.middle_flow(middle_out)
        
        # exit flow
        exit_output = self.exit_flow(middle_out)
        
        # fully connected layers
        output = self.fc(exit_output)

        return output
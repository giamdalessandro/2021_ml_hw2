import torch
import torch.nn as nn
import torch.nn.functional as F



class depthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution module, `modified` Xception configuration as default.
    """
    def __init__(self, n_in, n_out, kernel_size, padding, bias=False, mode: str="modified"):
        """
        - mode: 'original' or 'modified' depthwise separable convolution.
        """
        super(depthwiseSeparableConv, self).__init__()
        self.mode = mode
        if self.mode == "modified":
            self.pointwise = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias)
            self.depthwise = nn.Conv2d(n_out, n_out, kernel_size=kernel_size, padding=padding, groups=n_out, bias=bias)
        else:
            self.depthwise = nn.Conv2d(n_in, n_in, kernel_size=kernel_size, padding=padding, groups=n_in, bias=bias)
            self.pointwise = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias)

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
    def __init__(self, input_channel, widen_factor: int):
        super(EntryFlow, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channel, 4*widen_factor, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*widen_factor),
            nn.ReLU(True),
            
            nn.Conv2d(4*widen_factor, 8*widen_factor, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*widen_factor),
            nn.ReLU(True)
        )
        
        self.block_2 = nn.Sequential(
            depthwiseSeparableConv(8*widen_factor, 16*widen_factor, 3, 1),
            nn.BatchNorm2d(16*widen_factor),
            nn.ReLU(True),
            
            depthwiseSeparableConv(16*widen_factor, 16*widen_factor, 3, 1),
            nn.BatchNorm2d(16*widen_factor),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_2_residual = nn.Conv2d(8*widen_factor, 16*widen_factor, kernel_size=1, stride=2, padding=0)
        
        self.block_3 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(16*widen_factor, 32*widen_factor, 3, 1),
            nn.BatchNorm2d(32*widen_factor),
            
            nn.ReLU(True),
            depthwiseSeparableConv(32*widen_factor, 32*widen_factor, 3, 1),
            nn.BatchNorm2d(32*widen_factor),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_3_residual = nn.Conv2d(16*widen_factor, 32*widen_factor, kernel_size=1, stride=2, padding=0)
        
        self.block_4 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(32*widen_factor, 91*widen_factor, 3, 1),
            nn.BatchNorm2d(91*widen_factor),
            
            nn.ReLU(True),
            depthwiseSeparableConv(91*widen_factor, 91*widen_factor, 3, 1),
            nn.BatchNorm2d(91*widen_factor),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block_4_residual = nn.Conv2d(32*widen_factor, 91*widen_factor, kernel_size=1, stride=2, padding=0)

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
    def __init__(self, widen_factor: int):
        super(MiddleFlow, self).__init__()
        self.activation     = nn.ReLU(True)
        self.separable_conv = depthwiseSeparableConv(91*widen_factor, 91*widen_factor, 3, 1)
        self.normalization  = nn.BatchNorm2d(91*widen_factor)
    
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
    def __init__(self, widen_factor: int):
        super(ExitFlow, self).__init__()
        self.block_1 = nn.Sequential(
            nn.ReLU(True),
            depthwiseSeparableConv(91*widen_factor, 91*widen_factor, 3, 1),
            nn.BatchNorm2d(91*widen_factor),
            
            nn.ReLU(True),
            depthwiseSeparableConv(91*widen_factor, 128*widen_factor, 3, 1),
            nn.BatchNorm2d(128*widen_factor),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_1_residual = nn.Conv2d(91*widen_factor, 128*widen_factor, kernel_size=1, stride=2, padding=0)
        self.block_2 = nn.Sequential(
            depthwiseSeparableConv(128*widen_factor, 192*widen_factor, 3, 1),
            nn.BatchNorm2d(192*widen_factor),
            nn.ReLU(True),
            
            depthwiseSeparableConv(192*widen_factor, 256*widen_factor, 3, 1),
            nn.BatchNorm2d(256*widen_factor),
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
    def __init__(self, in_channels, num_classes: int=10, middle_rep: int=8, widen_factor: int=8):
        super(Xception, self).__init__()
        self.entry_flow  = EntryFlow(input_channel=in_channels, widen_factor=widen_factor)
        self.middle_flow = MiddleFlow(widen_factor=widen_factor)
        self.exit_flow   = ExitFlow(widen_factor=widen_factor)

        self.middle_rep  = middle_rep    # depth
        self.fc = nn.Sequential(         # fc top
            nn.Linear(256*widen_factor, num_classes) # 128*widen_factor),   
            #nn.Linear(128*widen_factor, num_classes)   
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    @staticmethod
    def get_classifiers():
        """
        Classifier is specified with a string of dash separated values, 
        representing `arch-depth-widenFactor` of the network. In this implementation 
        I consider the `depth` as the number of repetition of the `Middle Flow` Xception 
        module (8 times in the original paper).
        """
        return ['xc-8-8', 'xc-8-2', 'xc-4-4', 'xc-10-2']
    
    @classmethod
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):
        _, depth, widen = arch.split('-')
        cls_instance = cls(
            in_channels=input_channels, 
            num_classes=num_classes,
            middle_rep=int(depth), 
            widen_factor=int(widen)
        )
        return cls_instance

    def forward(self, x):
        # entry flow
        entry_out = self.entry_flow(x) 

        # middle flow, repeated 'depth' times
        middle_out = self.middle_flow(entry_out)
        for i in range(self.middle_rep-1):
            middle_out = self.middle_flow(middle_out)
        
        # exit flow
        exit_output = self.exit_flow(middle_out)
        
        # fully-connected top layers
        output = self.fc(exit_output)

        return output


if __name__ == "__main__":
    net = Xception(in_channels=3)
    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)

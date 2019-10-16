'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

# taken from https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models.cbam import CBAM


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(BasicBlock, self).__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(Bottleneck, self).__init__()
        self.use_cbam = use_cbam

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, reduction_ratio = 1, kernel_cbam = 3, use_cbam_block= False, use_cbam_class = False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.reduction_ratio = reduction_ratio
        self.kernel_cbam = kernel_cbam
        self.use_cbam_block = use_cbam_block
        self.use_cbam_class = use_cbam_class

        print(use_cbam_block, use_cbam_class)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if self.use_cbam_class:
            self.cbam = CBAM(n_channels_in = 512*block.expansion, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.reduction_ratio, self.kernel_cbam, self.use_cbam_block))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.use_cbam_class:
            out = out  + self.cbam(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out




def ResNet18(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
    print(kernel_cbam)
    return ResNet(
                BasicBlock, 
                [2,2,2,2], 
                reduction_ratio= reduction_ratio,
                kernel_cbam = kernel_cbam,
                use_cbam_block= use_cbam_block,
                use_cbam_class = use_cbam_class
                )

def ResNet34(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
    return ResNet(
        BasicBlock,
        [3,4,6,3],
        reduction_ratio= reduction_ratio, 
        kernel_cbam = kernel_cbam, 
        use_cbam_block= use_cbam_block,
        use_cbam_class = use_cbam_class
        )

def ResNet50(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
    return ResNet(
        Bottleneck, 
        [3,4,6,3], 
        reduction_ratio= reduction_ratio, 
        kernel_cbam = kernel_cbam, 
        use_cbam_block= use_cbam_block,
        use_cbam_class = use_cbam_class
        )

def ResNet101(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
    return ResNet(
        Bottleneck, 
        [3,4,23,3], 
        reduction_ratio= reduction_ratio, 
        kernel_cbam = kernel_cbam, 
        use_cbam_block= use_cbam_block,
        use_cbam_class = use_cbam_class
        )

def ResNet152(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
    return ResNet(
        Bottleneck, 
        [3,8,36,3], 
        reduction_ratio= reduction_ratio, 
        kernel_cbam = kernel_cbam, 
        use_cbam_block= use_cbam_block,
        use_cbam_class = use_cbam_class
        )

def ResNetk(k, reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False ):
    possible_depth = [18,34,50,101,152]
    assert k in possible_depth, "Choose a depth in {}".format(possible_depth)

    if k == 18:
        return ResNet18(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
    elif k == 34:
        return ResNet34(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
    elif k == 50:
        return ResNet50(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
    elif k == 101:
        return ResNet101(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
    elif k == 152:
        return ResNet152(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

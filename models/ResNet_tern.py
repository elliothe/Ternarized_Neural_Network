import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.nn import init

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class _quanFunc(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_quanFunc,self).__init__()
        self.tFactor = tfactor

    def forward(self, input):
        self.save_for_backward(input)
        max_w = input.abs().max()
        self.th = self.tFactor*max_w #threshold
        output = input.clone().zero_()
        self.W = input[input.ge(self.th)+input.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class quanConv2d(nn.Conv2d):

    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

class quanLinear(nn.Linear):

    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        output = F.linear(input, weight, self.bias)

        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return quanConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = quanConv2d(inplanes, planes, kernel_size=1, stride=1, 
                                padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv2 = quanConv2d(planes, planes, kernel_size=3, stride=stride, 
                                padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3 = quanConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, 
                                padding=0, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fp_fl=True, fp_ll=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if fp_fl:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = quanConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if fp_ll:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = quanLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quanConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
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


def resnet18b_ff_lf_tex1(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], fp_fl=True, fp_ll=True)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18b_fq_lq_tex1(num_classes=1000):
    model = ResNet(BasicBlock, [2, 2, 2, 2], fp_fl=False, fp_ll=False)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34b_ff_lf_tex1(num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], fp_fl=True, fp_ll=True)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet34b_fq_lq_tex1(num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], fp_fl=False, fp_ll=False)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50b_ff_lf_tex1(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], fp_fl=True, fp_ll=True)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50b_fq_lq_tex1(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3], fp_fl=False, fp_ll=False)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101b_ff_lf_tex1(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], fp_fl=True, fp_ll=True)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet101b_fq_lq_tex1(num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 23, 3], fp_fl=False, fp_ll=False)
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


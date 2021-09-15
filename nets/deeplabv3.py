import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

__all__ = ['DeepLabV3', 'DeepLabHead']


class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x
        return result


class DeepLabHead(nn.Sequential):

    def __init__(self, in_ch, num_classes):
        rates = [12, 24, 36]
        super(DeepLabHead, self).__init__(
            ASPPModule(in_ch, rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        modules = [nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                   nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ASPPPooling, self).__init__()
        self.module = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = self.module(x.mean((2, 3), keepdim=True))
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPModule(nn.Module):
    def __init__(self, in_ch, rates, out_ch=256):
        super(ASPPModule, self).__init__()
        self.module1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

        self.module2 = ASPPConv(in_ch, out_ch, rates[0])
        self.module3 = ASPPConv(in_ch, out_ch, rates[1])
        self.module4 = ASPPConv(in_ch, out_ch, rates[2])

        self.module5 = ASPPPooling(in_ch, out_ch)

        self.project = nn.Sequential(nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout())

    def forward(self, x):
        out = torch.cat((self.module1(x),
                         self.module2(x),
                         self.module3(x),
                         self.module4(x),
                         self.module5(x)), dim=1)

        return self.project(out)


if __name__ == '__main__':
    model = DeepLabV3()

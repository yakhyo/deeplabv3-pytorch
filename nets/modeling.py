import torch
import torch.nn as nn
from nets.backbones import resnet, mobilenetv3
from collections import OrderedDict
from .deeplabv3 import DeepLabHead, DeepLabV3

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['deeplabv3_resnet50', 'deeplabv3_resnet101',
           'deeplabv3_mobilenet_v3_large']

model_urls = {
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
}


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        'return_layers'
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        return out


def _segm_model(name, backbone_name, num_classes, pretrained_backbone=True):
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        out_in_planes = 2048

    elif 'mobilenet_v3' in backbone_name:
        backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_in_planes = backbone[out_pos].out_channels

    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
    }
    classifier = model_map[name][0](out_in_planes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, **kwargs):
    if pretrained:
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model, arch_type, backbone, progress):
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)


def deeplabv3_resnet50(pretrained=False, progress=True, num_classes=21, **kwargs):
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True, num_classes=21, **kwargs):
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, **kwargs)


def deeplabv3_mobilenet_v3_large(pretrained=False, progress=True, num_classes=21, **kwargs):
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrained, progress, num_classes, **kwargs)


if __name__ == '__main__':
    model = _segm_model('deeplabv3', 'resnet50', 21, True)
    a = torch.randn((2, 3, 256, 256))
    print(model(a).shape)

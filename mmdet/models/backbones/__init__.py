from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .hrnet import HRNet
from .res2net import Res2Net
__all__ = ['ResNet', 'Res2Net','make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet']

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List

class IOPTFaces(nn.Module):
    def __init__(self, artifact_path, arq):
        super(IOPTFaces, self).__init__()

        self.model = self.load_model(artifact_path, arq)

    def load_model(self, artifact_path, arq):
        model = arq
        checkpoint = torch.load(artifact_path)  # , map_location=device)
        state_dict = checkpoint['net']
        model.load_state_dict(state_dict)
        return model

    def forward(self, img):
        self.model.eval()
        with torch.no_grad():
            output = self.model(img)
        return output

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

class MiniXception(nn.Module):
    def __init__(self, n_classes, last_activation='softmax'):
        super(MiniXception,self).__init__()

        first_convs = 8
        grow_rate = 2

        self.conv1 = BasicConv2d(1, first_convs, kernel_size=3)
        self.conv2 = BasicConv2d(first_convs, first_convs, kernel_size=3)

        self.deepresmodule1 = DeepwiseResidualModule(first_convs, 16)
        self.deepresmodule2 = DeepwiseResidualModule(16, 32)
        self.deepresmodule3 = DeepwiseResidualModule(32, 64)
        self.deepresmodule4 = DeepwiseResidualModule(64, 128)

        self.classifier = nn.Sequential(nn.Conv2d(128, n_classes, kernel_size=3, padding=1),
                                        nn.AdaptiveAvgPool2d((1, 1)))
        if last_activation is 'sigmoid':
            self.last_activation = nn.Sigmoid()
        elif last_activation is 'softmax':
            self.last_activation = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deepresmodule1(x)
        x = self.deepresmodule2(x)
        x = self.deepresmodule3(x)
        x = self.deepresmodule4(x)
        x = self.classifier(x)#.squeeze(-1).squeeze(-1)
        x = x.view((x.shape[0], -1))
        if hasattr(self, 'last_activation'):
            x = self.last_activation(x)
        return x

class DeepwiseResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DeepwiseResidualModule, self).__init__()
        self.residual = BasicConv2d(in_channels, out_channels,
                                    activation=None, kernel_size=1, stride=2)

        self.separable1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.separable2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, activation=None)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual(x)
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.pool(x)
        return torch.add(res, x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation: Any = 'relu',
                 kernel_size: int = 1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 **kwargs: Any) -> None:
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        if activation == 'relu':
            self.activation = nn.ReLU()

        # (2 * (output - 1) - input - kernel) * (1 / stride)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import pathlib
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    dummy = torch.ones([1,1,48,48]).cuda()

    model = MiniXception(len(CLASS_NAMES), last_activation='sigmoid').cuda()

    print(summary(model, input_size=(1, 48, 48 )))
    # model.load_state_dict(torch.load('./saved_models/IOPTFacial_sgd_balanced_checkpoint.pth.tar')['state_dict'])



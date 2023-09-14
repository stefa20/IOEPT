import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List
from  torchsummary import summary

class ResNet18(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size, include_top=True):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features

        if include_top:
            self.resnet18.fc = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                nn.Softmax()
            )

    def forward(self, x):
        x = self.resnet18(x)
        return x

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size, include_top=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features

        if include_top:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                # nn.Softmax()
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class MobileNet(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, classCount, include_top=True):
        super(MobileNet, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=include_top)
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, classCount),
            # nn.Sigmoid()
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x

class InceptionVAE(nn.Module):

    def __init__(self, latent_size):
        super(InceptionVAE, self).__init__()
        self.encoder = InceptionEncoder(latent_size=latent_size)
        self.decoder = InceptionDecoder(latent_size=latent_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        p_x = self.decoder(z)
        return p_x, mu, logvar

class InceptionEncoder(nn.Module):

    def __init__(
            self,
            latent_size,
            model_blocks: Optional[List[Callable[..., nn.Module]]] = None,
    ) -> None:
        super(InceptionEncoder, self).__init__()
        if model_blocks is None:
            model_blocks = [BasicConv2d, InceptionBlock, nn.MaxPool2d]

        assert len(model_blocks) == 3
        conv_block = model_blocks[0]
        inception_a =model_blocks[1]
        pool_block = model_blocks[2]

        self.inception_block1 = inception_a(3, pool_features=8,
                                            n_features_branch=8)
        self.max_pool1 = pool_block(kernel_size=2)

        self.inception_block2 = inception_a(32, pool_features=16,
                                            n_features_branch=16)
        self.max_pool2 = pool_block(kernel_size=2)

        self.inception_block3 = inception_a(64, pool_features= 32,
                                            n_features_branch=32)
        self.max_pool3 = pool_block(kernel_size=2)
        self.flat = nn.Flatten()

        self.encoder_mu = nn.Sequential(nn.Linear(128, latent_size))
        self.encoder_logvar = nn.Sequential(nn.Linear(128, latent_size))

    def encode(self, x):
        x = self.inception_block1(x)
        x = self.max_pool1(x)
        x = self.inception_block2(x)
        x = self.max_pool2(x)
        x = self.inception_block3(x)
        x = self.max_pool3(x)
        x = F.avg_pool2d(x, kernel_size=4)
        x = self.flat(x)
        return x#.unsqueeze(-1)

    def gaussian_param_projection(self, x):
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encode(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class InceptionDecoder(nn.Module):
    def __init__(
            self,
            latent_size,
            model_blocks: Optional[List[Callable[..., nn.Module]]] = None,
    ) -> None:
        super(InceptionDecoder, self).__init__()
        if model_blocks is None:
            model_blocks = [BasicConv2d, InceptionBlock, nn.Upsample]

        assert len(model_blocks) == 3
        conv_block = model_blocks[0]
        inception_a = model_blocks[1]
        upsamp_block = model_blocks[2]

        self.upsample_conv = nn.ConvTranspose2d(latent_size, 128, kernel_size= 4)

        self.inception_block1 = inception_a(128, pool_features=16,
                                            n_features_branch=16)
        self.upsample1 = upsamp_block(scale_factor=(4,4))

        self.inception_block2 = inception_a(64, pool_features=8,
                                            n_features_branch=8)
        self.upsample2 = upsamp_block(scale_factor=(2,2))

        self.last_conv2d = conv_block(32, out_channels=3, kernel_size=1)
        # self.max_pool3 = pool_block(kernel_size=2)

    def decode(self, x):
        x = self.upsample_conv(x)
        x = self.inception_block1(x)
        x = self.upsample1(x)
        x = self.inception_block2(x)
        x = self.upsample2(x)
        x = self.last_conv2d(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.decode(x.unsqueeze(dim=-1))

class InceptionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        n_features_branch: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionBlock, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, n_features_branch, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, n_features_branch, kernel_size=1)
        self.branch5x5_2 = conv_block(n_features_branch, n_features_branch, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, n_features_branch, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(n_features_branch, n_features_branch, kernel_size=3, padding=1)
        # self.branch3x3dbl_3 = conv_block(96, n_features_branch, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

if __name__ == '__main__':
    import numpy as np

    # encoder = InceptionEncoder(latent_size=20).cuda()
    dummy = torch.ones([4, 3,32,32]).cuda()
    # summary(model= encoder, input_data=(3, 32, 32))
    # output = encoder(dummy)
    # print(output[0].shape)
    #
    # decoder = InceptionDecoder(latent_size=20).cuda()
    # summary(model= decoder, input_data=(1, 20))
    # reconst = decoder(output)
    # print(reconst)

    cae = ResNet18(2).cuda()
    summary(model= cae, input_size=(3, 224, 224))
    output = cae(dummy)
    print(output[0].shape)

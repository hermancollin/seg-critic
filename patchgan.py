import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# TODO: ARGPARSE
# TODO: INIT NETWORK, LOSS, GPU SUPPORT
# TODO: INIT DATALOADER (network needs 256x256 input images)
# TODO: TRAINING LOOP
# TODO: |___ sample a mask and use its quality label to train D 
# no need for checkpoints: training should be fairly fast

def weights_init_normal(m):
    """Typical GAN weight initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class PatchGAN(nn.Module):
    # Note: for a receptive field of [70x70], input images are 256x256
    def __init__(self, in_channels=1):
        super(PatchGAN, self).__init__()

        def patchgan_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layer of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *patchgan_block(in_channels, 64, normalization=True),
            *patchgan_block(64, 128),
            *patchgan_block(128, 256),
            *patchgan_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
        )

        def forward(self, img):
            return self.model(img)


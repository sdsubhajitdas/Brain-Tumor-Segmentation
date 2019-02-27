import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNet(nn.Module):
    """TODO: Method was actually 4 level depth but reduced to 3 level.

    This is the Pytorch version of U-Net Architecture.
    This is not the vanilla version of U-Net.
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597

    This network is modified to have only 3 blocks depth because of
    computational limitations. 
    The input and output of this network is of the same shape.
    Input Size of Network - (1,512,512)
    Output Size of Network - (1,512,512)
        Shape Format :  (Channel, Width, Height)
    """

    def __init__(self):
        """ Constructor for UNet class.
        Filters used at Encoding part of the network.
            Level 1: 64
            Level 2: 128
            Level 3: 256
        Filters used at Decoding part of the network.
            Level 2: 128 (128+256 --> 128)
            Level 3: 64 (64+128 --> 64)

        Each level has a general sequential series.
            Conv --> ReLU --> Conv --> ReLU
            Conv Layer Details:-
                Kernel Size = (3,3)
                Padding Size = (1,1)
        Methods used for level transition in 
            Encoding Part:
                MaxPool - Kernel Size (2,2)
            Decoding Part:
                Upsampling - Scale Factor 2, mode(bilinear)
        """
        super(UNet, self).__init__()

        # Encoding Part
        self.down_block_1 = self._get_double_conv_block(1, 64)
        self.down_block_2 = self._get_double_conv_block(64, 128)

        # Base Level
        self.down_block_3 = self._get_double_conv_block(128, 256)
        #self.base_block = self._get_double_conv_block(256,512)

        # Decoding Part
        #self.up_block_3 = self._get_double_conv_block(256+512,256)
        self.up_block_2 = self._get_double_conv_block(128+256, 128)
        self.up_block_1 = self._get_double_conv_block(64+128, 64)

        # Output Part
        self.output_block = self._get_output_conv_block(64, 1)

    def forward(self, x):
        """ Method for forward propagation in the network.
        TODO: Commented out code lines are for 4th Level 
        Parameters:
            x(torch.Tensor): Input for the network.

        Returns:
            output(torch.Tensor): Output after the forward propagation 
                                    of network on the input.
        """

        # Encoding part.
        #   Level 1
        down_block_1 = self.down_block_1(x)
        max_pool = F.max_pool2d(down_block_1, kernel_size=2)

        #   Level 2
        down_block_2 = self.down_block_2(max_pool)
        max_pool = F.max_pool2d(down_block_2, kernel_size=2)

        #   Base Level
        down_block_3 = self.down_block_3(max_pool)
        #max_pool = F.max_pool2d(down_block_3,kernel_size=2)

        #base_block = self.base_block(max_pool)

        #up_block_3 = F.interpolate(base_block,scale_factor=2,mode='bilinear',align_corners=True)
        #up_block_3 = torch.cat((down_block_3,up_block_3),dim=1)
        #up_block_3 = self.up_block_3(up_block_3)

        # Decoding Part
        #   Level 2
        #up_block_2 = F.interpolate(up_block_3,scale_factor=2,mode='bilinear',align_corners=True)
        up_block_2 = F.interpolate(
            down_block_3, scale_factor=2, mode='bilinear', align_corners=True)
        up_block_2 = torch.cat((down_block_2, up_block_2), dim=1)
        up_block_2 = self.up_block_2(up_block_2)

        #   Level 1
        up_block_1 = F.interpolate(
            up_block_2, scale_factor=2, mode='bilinear', align_corners=True)
        up_block_1 = torch.cat((down_block_1, up_block_1), dim=1)
        up_block_1 = self.up_block_1(up_block_1)

        # Output Part
        output = self.output_block(up_block_1)

        return output

    def _get_double_conv_block(self, in_channels, out_channels):
        """ Returns a sequential block of
        Conv --> ReLU --> Conv --> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU())

    def _get_output_conv_block(self, in_channels, out_channels):
        """Returns a Convolution Layer with Kernel Size (1,1)
        and no padding for output layer.
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
        """ Get the summary of the network in a chart like form
        with name of layer size of the inputs and parameters 
        and some extra memory details.
        This method uses the torchsummary package.
        For more information check the link.
        Link :- https://github.com/sksq96/pytorch-summary

        Parameters:
            input_size(tuple): Size of the input for the network in
                                 format (Channel, Width, Height).
                                 Default: (1,512,512)
            batch_size(int): Batch size for the network.
                                Default: -1
            device(str): Device on which the network is loaded.
                            Device can be 'cuda' or 'cpu'.
                            Default: 'cuda'

        Returns:
            A printed output for IPython Notebooks.
            Table with 3 columns for Layer Name, Input Size and Parameters.
            torchsummary.summary() method is used.
        """
        return summary(self, input_size, batch_size, device)

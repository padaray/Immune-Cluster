import torch
from torch import nn

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, base_kernal = 4):
        """ base_kernal(int) = 4: power(2, base_kernal) is the base number of kernal in each layer 
        """
        super().__init__()

        self.conv1 = self.conv_block(in_channels, 2**(base_kernal), 3, 1)
        self.maxpool1 = self.max_pool_layer()
        self.conv2 = self.conv_block(2**(base_kernal), 2**(base_kernal+1), 3, 1)
        self.maxpool2 = self.max_pool_layer()
        self.conv3 = self.conv_block(2**(base_kernal+1), 2**(base_kernal+2), 3, 1)
        self.maxpool3 = self.max_pool_layer()
        self.conv4 = self.conv_block(2**(base_kernal+2), 2**(base_kernal+3), 3, 1)
        self.maxpool4 = self.max_pool_layer()
        
        self.upconv4 = self.conv_block(2**(base_kernal+3), 2**(base_kernal+4), 3, 1)  #(in_channels, out_channels, kernel_size, padding)
        self.ConvT4 = self.expansive_layer(2**(base_kernal+4), 2**(base_kernal+3))
        self.upconv3 = self.conv_block(2**(base_kernal+4), 2**(base_kernal+3), 3, 1)
        self.ConvT3 = self.expansive_layer(2**(base_kernal+3), 2**(base_kernal+2))
        self.upconv2 = self.conv_block(2**(base_kernal+3), 2**(base_kernal+2), 3, 1)
        self.ConvT2 = self.expansive_layer(2**(base_kernal+2), 2**(base_kernal+1))
        self.upconv1 = self.conv_block(2**(base_kernal+2), 2**(base_kernal+1), 3, 1)
        self.ConvT1 = self.expansive_layer(2**(base_kernal+1), 2**(base_kernal))
        self.upconv0 = self.conv_block(2**(base_kernal+1), 2**(base_kernal), 3, 1)
        self.output = self.output_block(2**(base_kernal), out_channels, 3, 1)
  
    def __call__(self, x):
        # contracting path
        conv1 = self.conv1(x)
        conv1_pool = self.maxpool1(conv1)
        conv2 = self.conv2(conv1_pool)
        conv2_pool = self.maxpool2(conv2)
        conv3 = self.conv3(conv2_pool)
        conv3_pool = self.maxpool3(conv3)
        conv4 = self.conv4(conv3_pool)
        conv4_pool = self.maxpool4(conv4)
       
        # expansive path
        upconv4 = self.upconv4(conv4_pool)
        ConvT4 = self.ConvT4(upconv4)
        upconv3 = self.upconv3(torch.cat([ConvT4, conv4], 1))
        ConvT3 = self.ConvT3(upconv3)
        upconv2 = self.upconv2(torch.cat([ConvT3, conv3], 1))
        ConvT2 = self.ConvT2(upconv2)
        upconv1 = self.upconv1(torch.cat([ConvT2, conv2], 1))
        ConvT1 = self.ConvT1(upconv1)
        upconv0 = self.upconv0(torch.cat([ConvT1, conv1], 1))
        output = self.output(upconv0)
        #output = torch.sigmoid(output)

        return output

    def conv_block(self, in_channels, out_channels, kernel_size, padding):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )   
        return conv   

    def max_pool_layer(self):
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        return max_pool

    def expansive_layer(self, in_channels, out_channels):
        #up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        #up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return up_conv

    def output_block(self, in_channels, out_channels, kernel_size, padding):
        output_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        )   
        return output_conv

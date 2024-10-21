import torch
import torch.nn as nn
import logging
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n0 = 8):
        super(UNet, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, n0)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(n0, 2*n0)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(2*n0, 4*n0)
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv4 = ConvBlock(4*n0, 8*n0)
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv5 = ConvBlock(8*n0, 16*n0)

        self.upconv1 = ConvBlock(24*n0, 8*n0)
        self.upconv2 = ConvBlock(12*n0, 4*n0)
        self.upconv3 = ConvBlock(6*n0, 2*n0)
        self.upconv4 = ConvBlock(3*n0, 1*n0)
        self.conv6 = nn.Conv3d(n0, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        conv1_out = self.conv1(x)
        pool1_out = self.pool1(conv1_out)
        conv2_out = self.conv2(pool1_out)
        pool2_out = self.pool2(conv2_out)
        conv3_out = self.conv3(pool2_out)
        pool3_out = self.pool3(conv3_out)
        conv4_out = self.conv4(pool3_out)
        pool4_out = self.pool4(conv4_out)
        conv5_out = self.conv5(pool4_out)
        
        upsample1 = F.interpolate(conv5_out, size=conv4_out.shape[2:], mode='nearest')
        concat1_out = torch.cat([conv4_out, upsample1], dim=1)
        upconv1 = self.upconv1(concat1_out)
        upsample2 = F.interpolate(upconv1, size=conv3_out.shape[2:], mode='nearest')
        concat2_out = torch.cat([conv3_out, upsample2], dim=1)
        upconv2 = self.upconv2(concat2_out)
        upsample3 = F.interpolate(upconv2, size=conv2_out.shape[2:], mode='nearest')
        concat3_out = torch.cat([conv2_out, upsample3], dim=1)
        upconv3 = self.upconv3(concat3_out)
        upsample4 = F.interpolate(upconv3, size=conv1_out.shape[2:], mode='nearest')
        concat4_out = torch.cat([conv1_out, upsample4], dim=1)
        upconv4 = self.upconv4(concat4_out)
        conv6_out = self.conv6(upconv4)

        #print("Size of output before softmax", conv6_out.shape)
        # ====================================
        # print shapes at various layers in the network
        #====================================
        #logging.info('=======================================================')
        #logging.info('Details of the segmentation CNN architecture')
        #logging.info('=======================================================')
        #logging.info('Shape of input: ' + str(x.shape))        
        #logging.info('Shape after 1st max pooling layer: ' + str(pool1_out.shape))
        #logging.info('Shape after 2nd max pooling layer: ' + str(pool2_out.shape))        
        #logging.info('Shape after 3rd max pooling layer: ' + str(pool3_out.shape))        
        #logging.info('Shape after 4th max pooling layer: ' + str(pool4_out.shape))            
        #logging.info('=======================================================')
        #logging.info('Before each maxpool, there are 2 conv blocks.')
        #logging.info('Each conv block consists of conv3d (k=3), followed by BN, followed by relu.')
        #logging.info('=======================================================')
        #logging.info('Shape of the bottleneck layer: ' + str(conv5_out.shape))            
        #logging.info('=======================================================')
        #logging.info('Shape after 1st upsampling block: ' + str(upconv6_out.shape))            
        #logging.info('Shape after 2nd upsampling block: ' + str(upconv7_out.shape))     
        #logging.info('Shape after 3rd upsampling block: ' + str(upconv8_out.shape))     
        #logging.info('Shape after 4rd upsampling block: ' + str(upconv9_out.shape)) 
        #logging.info('=======================================================')
        #logging.info('Each upsampling block consists of bilinear upsampling, followed by skip connection, followed by 2 conv blocks.')
        #logging.info('=======================================================')
        #logging.info('Shape of output (before softmax): ' + str(conv6_out.shape)) 
        #logging.info('=======================================================')
        return conv6_out
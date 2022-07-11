# パッケージのimport
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, bias=False)
        self.conv_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, bias=False)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self,x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x
    
    
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.Double_Conv = DoubleConv(in_channels, out_channels)
    def forward(self,x):
        x = self.max_pool(x)
        x = self.Double_Conv(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = False):
        super(Up, self).__init__()
        self.up_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        # self.Double_Conv = DoubleConv(out_channels, out_channels)
        
    def forward(self, x):
        x = self.up_trans(x)
        # x = self.Double_Conv(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,input_size,bilinear = False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_size = input_size
        self.Double_Conv_0 = DoubleConv(n_channels,64)
        self.Down_1 = Down(64,128)
        self.Down_2 = Down(128,256)
        self.Down_3 = Down(256,512)
        self.Down_4 = Down(512,1024)
        # self.Up_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.Up_1 = Up(1024,512)
        self.Double_Conv_1 = DoubleConv(1024,512)
        
        self.Up_2 = Up(512,256)
        self.Double_Conv_2 = DoubleConv(512,256)
        
        self.Up_3 = Up(256,128)
        self.Double_Conv_3 = DoubleConv(256,128)
        
        self.Up_4 = Up(128,64)
        self.Double_Conv_4 = DoubleConv(128,64)
        
        self.output = nn.Conv2d(64,self.n_classes, kernel_size = 1, bias = False)
        
        
    def crop_image(self, x1, x2):
        delta = x1.size()[2] - x2.size()[2]
        if delta % 2 != 0:
            delta = delta // 2
            x1 = x1[:, :, delta:x1.size()[2] - delta - 1, delta:x1.size()[3] - delta - 1]
        else:
            delta = delta//2
            x1 = x1[:, :, delta:x1.size()[2] - delta, delta:x1.size()[3] - delta]

        return x1
    
    
    def forward(self,x):
        x1 = self.Double_Conv_0(x)
        x2 = self.Down_1(x1)
        x3 = self.Down_2(x2)
        x4 = self.Down_3(x3)
        x5 = self.Down_4(x4)
        x6 = self.Up_1(x5)
        x7 = torch.cat([x6,self.crop_image(x4,x6)],dim = 1)
        x7 = self.Double_Conv_1(x7)
        
        x7 = self.Up_2(x7)
        x7 = torch.cat([x7,self.crop_image(x3,x7)],dim = 1)
        x7 = self.Double_Conv_2(x7)
        
        x7 = self.Up_3(x7)
        x7 = torch.cat([x7,self.crop_image(x2,x7)],dim = 1)
        x7 = self.Double_Conv_3(x7)
        
        x7 = self.Up_4(x7)
        x7 = torch.cat([x7,self.crop_image(x1,x7)],dim = 1)
        x7 = self.Double_Conv_4(x7)
        
        x7 = F.interpolate(x7,size=(self.input_size,self.input_size), mode="bilinear", align_corners=True )
        output = self.output(x7)

        return output
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegModel(nn.Module):
    def __init__(self,in_ch=14):
        super().__init__()
        self.in_ch = in_ch
        self.feat1 = nn.Sequential(nn.Conv3d(in_ch,16,3,padding=1,stride=1),nn.InstanceNorm3d(16),nn.ReLU(),\
                                  nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,32,3,padding=1))
        self.feat2 = nn.Sequential(nn.Conv3d(32,64,3,padding=1,stride=2),nn.InstanceNorm3d(64),nn.ReLU(),\
                                  nn.Conv3d(64,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),nn.Conv3d(64,64,3,padding=1))
        self.feat3 = nn.Sequential(nn.Conv3d(64,64,3,padding=1,stride=1),nn.InstanceNorm3d(64),nn.ReLU(),\
                                  nn.Conv3d(64,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),nn.Conv3d(64,64,3,padding=1))
        #self.reg1 = nn.Sequential(nn.Conv3d(64,64,3,padding=1,stride=2),nn.InstanceNorm3d(64),nn.ReLU(),\
        #                          nn.Conv3d(64,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),nn.Conv3d(64,3,3,padding=1))
        self.reg2 = nn.Sequential(nn.Conv3d(128,64,3,padding=1,stride=2),nn.InstanceNorm3d(64),nn.ReLU(),\
                                  nn.Conv3d(64,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),nn.Conv3d(64,3,3,padding=1))
        self.reg3 = nn.Sequential(nn.Conv3d(128,64,3,padding=1,stride=2),nn.InstanceNorm3d(64),nn.ReLU(),\
                                  nn.Conv3d(64,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),nn.Conv3d(64,3,3,padding=1))

    def forward(self, x,y,level=1):
        H,W,D = x.shape[-3:]
        grid = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D))

        x3 = self.feat3(self.feat2(self.feat1(x)))
        y3 = self.feat3(self.feat2(self.feat1(y)))
        disp3 = torch.tanh(self.reg3(torch.cat((x3,y3),1)))*.25
        #print(disp3.shape)
        #smooth
        disp3 = F.avg_pool3d(F.avg_pool3d(F.interpolate(disp3,scale_factor=2,mode='trilinear'),3,stride=1,padding=1),5,stride=1,padding=2)
        disp3 = F.avg_pool3d(F.avg_pool3d(F.interpolate(disp3,scale_factor=2,mode='trilinear'),3,stride=1,padding=1),5,stride=1,padding=2)
        if(level==2):
            y3_ = F.grid_sample(y,grid+disp3.permute(0,2,3,4,1))
            x2 = self.feat2(self.feat1(x))
            y2 = self.feat2(self.feat1(y3_))
            #smooth
            disp2 = torch.tanh(self.reg2(torch.cat((x2,y2),1)))*.25
            disp2 = F.avg_pool3d(F.avg_pool3d(F.interpolate(disp2,scale_factor=2,mode='trilinear'),3,stride=1,padding=1),5,stride=1,padding=2)
            disp2 = F.avg_pool3d(F.avg_pool3d(F.interpolate(disp2,scale_factor=2,mode='trilinear'),3,stride=1,padding=1),5,stride=1,padding=2)
            disp3 += disp2
        
    
        return disp3


class ShufflePermutation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Flatten(0,1),nn.Unflatten(0,(-1,1)),nn.Conv3d(1,16,3,padding=1,stride=2),nn.InstanceNorm3d(16),nn.ReLU(),\
                                   nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),\
                                   nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),\
                                   nn.Upsample(scale_factor=2,mode='trilinear'),\
                                   nn.Conv3d(16,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,1))
    def forward(self, x):
        B,C = x.shape[:2]
        x = torch.max(nn.Unflatten(0,(B,C))(self.conv1(x)),1).values
        return self.conv2(x)
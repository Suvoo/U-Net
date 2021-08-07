import torch
import torch.nn as nn  

# print(torch.cuda.is_available())

def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c,kernel_size = 3),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c,kernel_size = 3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_image(tensor,target):
    target_size = target.size()[2]
    tensor_size = tensor.size()[2]

    diff = tensor_size - target_size
    diff = diff // 2

    return tensor[:,:,diff:tensor_size-diff,diff:tensor_size-diff]



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(1,64)
        self.down_conv2 = double_conv(64,128)
        self.down_conv3 = double_conv(128,256)
        self.down_conv4 = double_conv(256,512)
        self.down_conv5 = double_conv(512,1024)

        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(1024,512)

        self.up_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(512,256)

        self.up_trans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3= double_conv(256,128)

        self.up_trans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128,64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)




    def forward(self, image):
        #encoder
        x1 = self.down_conv1(image) #op
        x2 = self.max_pool_2x2(x1)

        x3 = self.down_conv2(x2) #op
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv3(x4) #op
        x6 = self.max_pool_2x2(x5)

        x7 = self.down_conv4(x6) #op
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv5(x8) #op

        #decoder
        x = self.up_trans1(x9)
        y = crop_image(x7, x)
        x =  self.up_conv1(torch.cat([x,y],1))

        x = self.up_trans2(x)
        y = crop_image(x5, x)
        x =  self.up_conv2(torch.cat([x,y],1))

        x = self.up_trans3(x)
        y = crop_image(x3, x)
        x =  self.up_conv3(torch.cat([x,y],1))

        x = self.up_trans4(x)
        y = crop_image(x1, x)
        x =  self.up_conv4(torch.cat([x,y],1))

        x = self.out(x)
        return x

        print(x1.size())
        print(x3.size())
        print(x5.size())
        print(x7.size())
        print(x9.size())

        # print(x.size())
        print(y.size())

        print(x.size())

        # check x7 and x size. so either 1.crop 2.padding

        # batch size, channel, height, width
if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = UNet()
    print(model)
    print(model(image))




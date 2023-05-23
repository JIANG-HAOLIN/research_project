import torch
import torch.nn as nn
import torch.nn.functional as F
# import config


# opt = config.read_arguments(train=True)

class trans_encoder(torch.nn.Module):
    def __init__(self,opt=None,dim=1024):
        super(trans_encoder,self).__init__()
        self.encoder_list = torch.nn.ModuleList()
        num_layers = 3 if opt==None else opt.num_layers
        for i in range(num_layers) :
            self.encoder_list.append(torch.nn.TransformerEncoderLayer(d_model=dim,
                                                                         nhead=8,)
                                     )

    def forward(self,input):
        out = []
        for layer in self.encoder_list:
            input = layer(input)
            x = input.permute(0, 2, 1).contiguous()
            out.append(x)
        return out




class Img2Token(nn.Module):

    def __init__(self, input_size=256, patch_size=16, dim=1024, padding=0, img_channels=35):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, 2*input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * ((input_size[1] + padding[1] * 2)  // patch_size[1])
        self.posemb = nn.Parameter(torch.randn(n_patches, dim)) ##512,1024

    def forward(self, data):
        x = data
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C通道数 * p * p, L一张图片几个patch)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc(x) + self.posemb.unsqueeze(0)
        return x

def Token2Img(input,input_size=256, patch_size=16, dim=1024, padding=0, img_channels=35):
    output = []
    for i in input:
        out = F.fold(input=i,kernel_size=patch_size,padding=padding,stride=patch_size,output_size=(input_size,2*input_size))
        output.append(out)
    return output



# input_image = torch.randn(4,35,256,512)
# img2token = Img2Token()
# input = img2token(input_image)##(B,N_patches 512,dim 2048)
# encoder = trans_encoder()
# out = encoder(input)
# out = Token2Img(out)
# print(out[0].shape)


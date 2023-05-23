import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_decoder(nn.Module):
    def __init__(self,opt=None,dim=8,out_dim=512):
        super(conv_decoder,self).__init__()
        self.decoder_list = torch.nn.ModuleList()
        self.num_layers = 3 if opt==None else opt.num_layers
        for i in range(self.num_layers) :
            if i == 0:
                self.decoder_list.append(torch.nn.Conv2d(in_channels=dim,out_channels=out_dim,kernel_size=1)
                                     )
            else:
                self.decoder_list.append(torch.nn.Conv2d(in_channels=dim+out_dim,out_channels=out_dim,kernel_size=1)
                                     )
    def forward(self,input_list):
        out_list = []
        for index,layer in enumerate(self.decoder_list):
            if index == 0 :
                input = input_list[self.num_layers-index-1]
                out = layer(input)
            else:
                input = torch.cat((input_list[self.num_layers-index-1],out),dim=1)
                out = layer(input)
            out_list.append(out)
        return out,out_list

# decoder = conv_decoder()
# input = [torch.randn(4, 8, 256, 512) for i in range(4)]
# out,_ = decoder(input)
# print(out.shape)
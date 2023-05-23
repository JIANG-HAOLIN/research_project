from models.trans_encoder import trans_encoder,Img2Token,Token2Img
from models.conv_decoder import conv_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder_decoder(nn.Module):
    def __init__(self,opt=None,dim=2048):
        super(encoder_decoder,self).__init__()
        self.opt = opt
        self.tokenizer = Img2Token(dim=dim)
        self.encoder = trans_encoder(self.opt,dim=dim)
        dim_de = (dim//1024)*4
        self.decoder = conv_decoder(self.opt,dim=dim_de)
    def forward(self,input):
        input_token = self.tokenizer(input)
        out_token = self.encoder(input_token)
        out_img = Token2Img(out_token)
        out,out_list = self.decoder(out_img)
        return out,out_list

# a = encoder_decoder()
# label = torch.randn(4,35,256,512)
# out,out_list = a(label)
# print(out.shape)
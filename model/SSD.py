import torch
import torch.nn as nn
from einops import rearrange
import math
from .MRU import mru

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class MRU(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, spatial_size,input_text_len, embed_dim,embed_dim1):

        super(MRU, self).__init__()

        self.spatial_size = spatial_size
        self.IGTI = mru(in_channels, in_channels, embed_dim,embed_dim,embed_dim, num_heads=8)


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        x = rearrange(x,'B C H W -> B (H W) C')

        out, text = self.IGTI(x, txt)
        out = rearrange(out,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        return out,text



class SAU(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, spatial_size,input_text_len, embed_dim,embed_dim1):

        super(SAU, self).__init__()

        self.spatial_size = spatial_size

        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,embed_dim1),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(embed_dim,max_len=input_text_len)

    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''

        x = rearrange(x,'B C H W -> B (H W) C')
        # Cross-Attention
        
        out,_ = self.cross_attn(query=self.vis_pos(x),
                                   key=self.txt_pos(txt),
                                   value=txt)
        out = rearrange(out,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        return out,txt

if __name__ == "__main__":
   

    model = MRU(256, 1, 14, 1, 256,256//2)

    model1 = SAU(256, 1, 14, 1, 256,256//2)

    x = torch.randn(1, 256, 14,14)
    text_feat = torch.randn(1, 1, 256)

    out, l_new  = model(x, text_feat)
    out1, l_new1   = model1(x, text_feat)

    print(out.shape)
    print(l_new.shape)

    print(out1.shape)
    print(l_new1.shape)

import torch
from torch import nn
from torch.nn import functional as F

class mru(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(mru, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                        #  nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(dropout)
                                         )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=dim,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                        # nn.GELU(),
                                        nn.ReLU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l):
        x = x.permute(0, 2, 1)
        l = l.permute(0, 2, 1)
        vis = self.vis_project(x)  # (B, dim, H*W)
        lang, l_new = self.image_lang_att(x, l)  # (B, dim, 1)
        mm = torch.mul(vis, lang)
        mm = mm+vis
        mm = self.project_mm(mm)  # (B, dim, H*W)
        mm = mm.permute(0, 2, 1)
        l_new = l_new.permute(0, 2, 1)


        return mm, l_new


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.f_value2 = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        # Out projection
        self.W1 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        self.W2 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.W3 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.value_channels, kernel_size=1, stride=1),
        )
        # self.project_emb = nn.Sequential(
        #     nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        # )
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)

    def forward(self, x, l):

        B, C, L = x.shape

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        value2 = self.f_value2(l)

        n_l = value2.size(-1)
        query = query.reshape(B, L, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        value2 = value2.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        pool_map = nn.AdaptiveAvgPool2d([1, n_l])(sim_map)  # (B,num_heads,1,N_l)

        l_new = (pool_map * value2).reshape(B, self.value_channels, n_l)  # (B,C,N_l)

        l_new_2 = self.W2(l_new)



        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, L, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W1(out)  # (B, value_channels, HW)
        return out, l_new_2



if __name__ == "__main__":
    model = mru(96, 96, 768,768,768, num_heads=8)
    x = torch.randn(1, 56*56, 96)
    l_feat = torch.randn(1, 24, 768)
    l_mask = torch.zeros(1, 20, 1)
    emb = torch.randn(1, 1, 768)
    out, l_new, word_new = model(x, l_feat, emb)
    print(out.shape)
    print(l_new.shape)
    print(word_new.shape)

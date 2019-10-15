import torch.nn as nn   
import torch
import torch.nn.functional as F



class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        return fp


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")

        
        return x

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out

def main():
    # ca = CBAM() 


    f = torch.FloatTensor([
        [
            [[1,1,1], [1,2,1], [1,1,1]],
            [[2,2,2], [2,3,2], [2,2,2]],
            [[3,3,3], [3,4,3], [3,3,3]]
        ]
    ])

    sa = SpatialAttention()
    sa(f)
    # cbam = CBAM(f.size()[1],1) 


    # fp = cbam(f)
    # print(f)
    # print(fp)
    


if __name__ == "__main__":
    main()
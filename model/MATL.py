import torch
from torch import nn
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=128, dropout=0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.attn_weights = None

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ChannelAttention_seq(nn.Module):
    def __init__(self, channel=64, ratio=8):
        super(ChannelAttention_seq, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, F):
        b, c, _, _ = F.size()
        F_avg = self.shared_layer(self.avg_pool(F).reshape(b, c))
        F_max = self.shared_layer(self.max_pool(F).reshape(b, c))
        M = self.sigmoid(F_avg + F_max).reshape(b, c, 1, 1)
        return F * M

class ChannelRecombination(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelRecombination, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class MATL(nn.Module):
    def __init__(self):
        super(MATL, self).__init__()
        self.conv_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(20, 17), stride=(1, 1), padding=(0, 8)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.conv_seq_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(20, 11), stride=(1, 1), padding=(0, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.conv_seq_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(20, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256)
        )

        self.attention_seq = ChannelAttention_seq(channel=448, ratio=64)

        self.conv_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 17), stride=(1, 1), padding=(0, 8)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.conv_shape_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 11), stride=(1, 1), padding=(0, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.conv_shape_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256)
        )

        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))

        self.transformer_shape = Transformer(101, 8, 8, 128, 128, 0.1)
        self.lstm = nn.LSTM(49, 21, 6, bidirectional=True, batch_first=True, dropout=0.2)
        self.channelrecombination = ChannelRecombination(448, 224)

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=448, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, seq, shape):
        seq = seq.float()
        seq_out1 = self.conv_seq_1(seq)
        seq_out2 = self.conv_seq_2(seq)
        seq_out3 = self.conv_seq_3(seq)
        seq_concat = torch.cat((seq_out1, seq_out2, seq_out3), dim=1)
        seq_concat = nn.functional.adaptive_max_pool2d(seq_concat, (1, 42))
        seq_atten = self.attention_seq(seq_concat)  # 32 448 1 101

        shape = shape.squeeze(1).float()
        encoder_shape_output = self.transformer_shape(shape)
        encoder_shape_output = encoder_shape_output.unsqueeze(1)  # 32*1*5*101
        shape_out1 = self.conv_shape_1(encoder_shape_output)
        shape_out2 = self.conv_shape_2(encoder_shape_output)
        shape_out3 = self.conv_shape_3(encoder_shape_output)
        shape_concat = torch.cat((shape_out1, shape_out2, shape_out3), dim=1)
        pool_shape_1 = self.max_pooling_1(shape_concat)
        pool_shape_1 = pool_shape_1.squeeze(2)  # 32*128*42
        out_shape, _ = self.lstm(pool_shape_1)
        out_shape1 = out_shape.unsqueeze(2)

        seq_out = self.channelrecombination(seq_atten)
        shape_out = self.channelrecombination(out_shape1)

        output = self.output(torch.cat((seq_out, shape_out), dim=1))

        return output

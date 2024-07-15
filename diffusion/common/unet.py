from blocks import  BasicConvBlock, SelfAttention2dBlock, SinusoidalPositionEmb
import torch.nn as nn
import torch

class Unet(nn.Module):
    def __init__(self,
                in_channel : int, # in and out channel for unet is the same
                emb_channel : int,
                dim_scales : list=[1, 2, 4, 8]
                ):
        super().__init__()

        self.init_emb = nn.Conv2d(in_channel, emb_channel, 1)
        self.time_emb = SinusoidalPositionEmb(emb_channel)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # creates layer channels
        all_dims = (emb_channel, *[emb_channel * s for s in dim_scales])

        # build down sampling blocks
        for i, (cin, cout) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = i == len(all_dims) - 2
            self.down_blocks.extend(
                nn.ModuleList([
                    BasicConvBlock(cin, cin, emb_channel),
                    BasicConvBlock(cin, cin, emb_channel),
                    nn.Conv2d(cin, cout, 3, 2, 1) if not is_last else nn.Conv2d(cin, cout, 1)
                ])
            )

        # build up sampling blocks
        for i, (cin, cout, skipc) in enumerate(zip(all_dims[::-1][:-1], 
                                                   all_dims[::-1][1:], 
                                                   all_dims[:-1][::-1])):
            is_last = i == len(all_dims) - 2
            self.up_blocks.extend(
                nn.ModuleList([
                    # we have a input channel that considers the skip connections for upsampling
                    BasicConvBlock(cin + skipc, cin, emb_channel),
                    BasicConvBlock(cin + skipc, cin, emb_channel),

                    # we use transpose2d here 
                    nn.ConvTranspose2d(cin, cout, 2, 2) if not is_last else nn.Conv2d(cin, cout, 1)
                ])
            )

        self.mid_blocks = nn.ModuleList([
            BasicConvBlock(all_dims[-1], all_dims[-1], emb_channel),
            SelfAttention2dBlock(all_dims[-1]),
            BasicConvBlock(all_dims[-1], all_dims[-1], emb_channel)
        ])

        self.out_blocks = nn.ModuleList([
            BasicConvBlock(emb_channel, emb_channel, emb_channel),
            nn.Conv2d(emb_channel, in_channel, 1, bias=True)
        ])

    def forward(self, x, t):

        # build embeddings
        x = self.init_emb(x)
        t = self.time_emb(t)

        skip_conns = []
        residuals = x.clone()

        for b in self.down_blocks:
            if isinstance(b, BasicConvBlock):
                x = b(x, t)
                skip_conns.append(x)
            else:
                x = b(x)

        for b in self.mid_blocks:
            if isinstance(b, BasicConvBlock):
                x = b(x, t)
            else:
                x = b(x)

        for b in self.down_blocks:
            if isinstance(b, BasicConvBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = b(x, t)
            else:
                x = b(x)

        # residual connection
        x = x + residuals

        for b in self.out_blocks:
            if isinstance(b, BasicConvBlock):
                x = b(x, t)
            else:
                x = b(x)
        return x



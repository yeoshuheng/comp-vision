import torch
import torch.nn as nn
from typing import Optional

# Deep Residual Learning for Image Recognition
# https://arxiv.org/pdf/1512.03385

# Residual / Skip Connections

# this wraps around a nn block and enables the resid connections,
# solves training problem where the earlier layers in a deep network fails to learn,
# because gradient becomes smaller (each activation function accumulatively decreases the gradient)

class ResidualAddBlock(nn.Module):

    def __init__(self, block : nn.Module) -> None:
        super().__init__()
        self.block = block
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_ = self.block(x)
        return x_ + x
    
class AdjustedResidualAddBlock(nn.Module):
    """
    Similar to ResidualAddBlock but with additionally conv layer
    to adapt to different output additions. (skip layer)
    """
    def __init__(self, block : nn.Module, shortcut : Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_ = self.block(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + x_
    
# Deep Residual Learning for Image Recognition
# https://arxiv.org/pdf/1512.03385

# Bottleneck Blocks 

# Shortens training time by using cheaper operations.

# [3 x 3], [3 x 3] conv layers are replaced with [1 x 1], [3 x 3], [1 x 1] kernels.
# the first [1 x 1] 'reduces' the input, the [3 x 3] works on a smaller space. 
# The final [1 x 1] projects it back to the input size.

# input: B x C x H x W
# after reducer: B x C/r x H x W (where r = reduction ratio)
# after middle: B x C/r x H x W 
# after increaser: B x C x H x W

# Linear Bottlenecks

# The implementation remains the same except we remove the activation.
# The rationale is that ReLU reduces some values to 0, which destroys information.

class ResidualBlock(nn.Module):
    def __init__(self, in_channel : int, out_channel : int, reduction : int = 4):
        super().__init__()
        reduced_feature = in_channel // reduction

        self.reducer = nn.Conv2d(in_channels=in_channel, out_channels=reduced_feature, kernel_size=1)
        self.middle = nn.Conv2d(in_channels=reduced_feature, out_channels=reduced_feature, kernel_size=3)
        self.expander = nn.Conv2d(in_channels=reduced_feature, out_channels=out_channel, kernel_size=1)

        self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.activation = nn.ReLU()

        self.block = nn.Sequential([self.reducer, self.activation, 
                                    self.middle, self.activation,
                                    self.expander, self.activation])
        
        self.residual_block = AdjustedResidualAddBlock(self.block, 
                                                       shortcut=self.shortcut if in_channel != out_channel else None)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.residual_block(x)
    
# MobileNetV2: Inverted Residuals and Linear Bottlenecks
# https://arxiv.org/abs/1801.04381

# Inverted BottleNeck

# The rationale is that the [3x3] convolution greatly reduces the number of 
# channels, afterwards, the final [1x1] brings it back to input size.

# The motivation is that feature maps can exist in low dimension, but we lose too much information w non-linear
# activations in such low dimensions, so we bring it to a high dimension, apply the non-linear activation
# and re-encode it back to a low dimension.

# Depth-wise Seperable Conv

# Splint a normal [3x3] convolution into 2 convolutions that reduce the number of parameters.
# First:
#    depthwise convolution: applies a single filter per channel.
# Second:
#    pointwise convolution: builds features by computing linear combinations of the input channel.

# Consider input with size B x C x H_i x W_i
# In 2DConv, with N kernels of size C x H_k x W_k, our final feature map would be of size B x N x H_f x W_f

# Total FloPs:
# In one convolution operation: 1 x C x H_k x W_k
# We have to shift it across the image H_i x H_w times, we do this for all N kernels to get
# Total operations = H_i x H_w x N x C x H_k x W_k

# In depthwise, we conv. across a single channel first.
# Each kernel will be of size 1 x H_k x W_k, our final feature map would be of size B x C x H_f x W_f

# Total FloPs:
# In one convolution operation: 1 x H_k x W_k
# We do this across each image channel, hence M times.
# Total operations = H_i x H_w x C x H_k x W_k (Reduction from original 2DConv by N times)

# In pointwise, we conv. across all M channel at once but only with a 1 x 1 kernel.
# Each kernel will be of size M x 1 x 1, with N kernels our final feature map would be of size B x N x H_f x W_f

# Total FloPs:
# In one convolution operation: M x 1 x 1
# With N kernels, we get
# Total operations = H_i x W_i x M x N (reduction by H_k x W_k)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channel : int, out_channel : int, expansion : int):
        super().__init__()
        expanded_feature = in_channel * expansion

        self.expander = nn.Conv2d(in_channel, expanded_feature, kernel_size=1)
        self.middle = nn.Conv2d(expanded_feature, expanded_feature, kernel_size=3)
        self.reducer = nn.Conv2d(expanded_feature, out_channel, kernel_size=1)

        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.activation = nn.ReLU()

        self.block = nn.Sequential([
            self.expander, self.activation,
            self.middle, self.activation,
            self.reducer, self.activation,
        ])

        self.residual_block = AdjustedResidualAddBlock(self.block, 
                                                       shortcut=self.shortcut if out_channel != in_channel else None)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.residual_block(x)
    

    



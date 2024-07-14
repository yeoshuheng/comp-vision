import torch.nn as nn
import torch, math

### utils ###

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, 
                 d : int, 
                 num_heads : int):
        super().__init__()
        assert num_heads % d == 0 # head should be divisible by d!

        self.d = d
        self.num_heads = num_heads
        self.d_k = num_heads // d

        # k, q, v weights
        self.W_k = nn.Linear(d, d)
        self.W_q = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)

        # output transform
        self.W_o = nn.Linear(d, d)

    def join_head(self, x) -> torch.Tensor:
        """[B, H, S, F_H] => [B, S, H, F_H] => [B, S, F]"""
        batch_size, seq_length, _, _ = x.shape
        
        # call contiguous() to ensure memory allocation does not change.
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d)

    def split_head(self, x) -> torch.Tensor:
        """[B, S, F] => [B, S, H, F_H] => transpose => [B, H, S, F_H]"""
        batch_size, seq_length, d = x.shape

        # for each token (with dimension d), we split across our heads so that we can
        # conduct multi-head attention with the same cost as single-head.

        # [B, S, F] => [B, S, H, F_H] => [B, S, F_H, H]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def scaled_dot_pdt_attention(self, K, Q, V, mask=None) -> torch.Tensor:

        # scale down based on embedding dimension
        # too high embedding dimension will kill the training
        
        #  [B, H, S, F_H] x [B, H, F_H, S] => [B, H, S, S]
        attention = torch.matmul(Q.transpose(-2, -1), K) / math.sqrt(self.d_k) 
        if mask is not None:
            attention = attention.masked_fill(mask == 0, value= -1e9)

        attention_prob = torch.softmax(attention, dim=1) # dim = 1 implies probability seen column-wise

       # [B, H, S, S] x [B, H, S, F_H] => [B, H, S, F_H] 
        output = torch.matmul(attention_prob, V)

        return output
    
    def forward(self, K, Q, V, mask=None):
        K_x = self.W_k(K)
        Q_x = self.W_q(Q)
        V_x = self.W_v(V)

        K_x = self.split_head(K_x)
        Q_x = self.split_head(Q_x)
        V_x = self.split_head(V_x)

        V_o = self.scaled_dot_pdt_attention(K_x, Q_x, V_x, mask=mask)
        V_o = self.join_head(V_o)
        V_o = self.W_o(V_o)

        return V_o
    
class PositionalEncoding(nn.Module):
    # read more: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    def __init__(self, 
                 d : int, 
                 max_seq_length : int = 512):
        super().__init__()
        self.d = d
        self.max_seq_length = max_seq_length

        # [S, D]
        padding = torch.zeros((max_seq_length, d))

        # [S,] => [S, 1] 
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)

        # multiply all odd indices by 0, 10000. is a hyperparam that determines
        # the sin / cos frequency, -log ensures that for higher values 
        # in the range, (ie. higher positions), our scale decreases.
        # low frequency => earlier tokens (words), high frequency => later tokens (words)

        # [D,] similar to binary operations, additionally for the element at the i-th position,
        # there exists a linear relationship to the i + k-th position.

        div_term = torch.exp(torch.arange(0, d, 2)).float() * -(math.log(10000.) / d)

        # we use sin / cos because they are orthogonal to each other, allowing us to
        # capture even more information

        padding[:, 0 :: 2] = torch.sin(position * div_term)
        padding[:, 1 :: 1] = torch.cos(position * div_term)

        # registering as buffer removes it from nn.parameters, hence the optimizer
        # will ignore it.
        self.register_buffer('padding', padding.unsqueeze(0))
    
        
    def forward(self, x):
        return x + self.padding[:, : x.shape[1]]
    
class PositionalWiseFeedforwardBlock(nn.Module):
    def __init__(self, 
                 d : int,
                 d_ff : int):
        super().__init__()

        # forward classifiers
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

### encoder block ###

class EncoderBlock(nn.Module):
    def __init__(self, 
                 d : int, 
                 num_heads : int, 
                 d_ff : int, 
                 dropout : float):
        super().__init__()

        # multi-head attention
        self.attention = MultiHeadAttentionBlock(d, num_heads)

        # fnn
        self.feed_forward = PositionalWiseFeedforwardBlock(d, d_ff)

        # norms
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # do
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.dropout(self.attention(x, x, x)) + x # skip
        x = self.norm1(x)

        x = self.dropout(self.feed_forward(x)) + x # skip2
        x = self.norm2(x)

        return x
    
### decoder block ###

class DecoderBlock(nn.Module):
    def __init__(self, 
                d : int,
                num_heads : int,
                d_ff : int,
                dropout : float):
        super().__init__()

        # self to self attention
        self.attention = MultiHeadAttentionBlock(d, num_heads)

        # self to input attention
        self.cross_attention = MultiHeadAttentionBlock(d, num_heads)

        self.feed_forward = PositionalWiseFeedforwardBlock(d, d_ff)

        # layer norms
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x : torch.Tensor, 
                enc_output : torch.Tensor, 
                source_mask, 
                tgt_mask) -> torch.Tensor:
        """
        :param x: source input
        :param enc_output: output from corresponding encoder
        :param source_mask: to mask encoder output
        :param tgt_mask: to mask decoder input
        """

        # attend to the target output sequence
        x = x + self.dropout(self.attention(x, x, x, mask=source_mask)) + x
        x = self.norm1(x)

        # takes in the input sequence (from the encoder) and shifted right outputs.
        # K = input, V = input
        # Q = output
        # essentially we use the output to adjust the K, V
        x = x + self.dropout(
            self.cross_attention(
                K=enc_output, 
                Q=x, 
                V=enc_output, mask=tgt_mask))
        x = self.norm2(x)

        x = self.dropout(self.feed_forward(x)) + x
        x = self.norm3(x)

        return x
    






        
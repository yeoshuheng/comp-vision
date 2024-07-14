from blocks import EncoderBlock, DecoderBlock, PositionalEncoding
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self,
                d : int=512,
                d_ff : int=2048,
                num_heads : int=8,
                src_vocab_size : int=5000,
                target_vocab_size : int=5000,
                max_seq_length : int=100,
                dropout : float=0.1,
                n_layers : int=6
                ):
        super().__init__()

        # embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, d)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d)
        self.positional_embedding = PositionalEncoding(d, max_seq_length)

        # decoder / encoder blocks
        self.encoders = nn.ModuleList([
            EncoderBlock(d, num_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(d, num_heads, d_ff, dropout) for _ in range(n_layers)
        ])


        self.fc = nn.Linear(d, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src : torch.Tensor, trgt : torch.Tensor):
        # [BS, MSL] => [BS, 1, 1, MSL]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
         # [BS, MSL] => [BS, 1, MSL, 1]
        trgt_mask = (src != 0).unsqueeze(1).unsqueeze(3)
        
        seq_len = trgt.shape[1]
        
        # [1, MSL, MSL]
        no_peek_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        
        # [BS, 1, MSL, 1] => broadcast [BS, 1, MSL, MSL]
        trgt_mask = trgt_mask & no_peek_mask # for the target we set a no-peek mask

        return src_mask, trgt_mask

    def forward(self, src : torch.Tensor, trgt : torch.Tensor) -> torch.Tensor:
        # build mask
        src_mask, trgt_mask = self.generate_mask(src, trgt)

        # convert to embedding -> add position
        src_embedding = self.encoder_embedding(src)
        src_embedding = self.positional_embedding(src_embedding)
        trgt_embedding = self.decoder_embedding(trgt)
        trgt_embedding = self.positional_embedding(trgt_embedding)

        src_embedding = self.dropout(src_embedding)
        trgt_embedding = self.dropout(trgt_embedding)

        # pass source / input into encoder
        enc_output = src_embedding
        for eb in self.encoders:
            enc_output = eb(enc_output, src_mask)

        # utilise the target sequence, which is what we previously output (trgt_embedding)
        # together with the input sequence (src_embedding)
        # masking only applied on the first layer
        # read: https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0
        
        dec_output = trgt_embedding
        for db in self.decoders:
            dec_output = db(dec_output, enc_output, src_mask, trgt_mask)

        output = self.fc(dec_output)

        return output
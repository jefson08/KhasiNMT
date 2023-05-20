import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rmsnorm_torch import RMSNorm
'''
SelfAttention Class
'''


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
        Split the embedding into different parts, the number of parts we are splitting is called heads,
        for e.g if embed_size=256 and heads=8 then we will split into 256/8=32 parts
        '''
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # // is integer division
        '''
        Say that embed_size=256 and we want to split into 7 parts then integer division would not be possible,
        therefore, use assert to throw out an assertion (error message)
        '''
        assert(self.head_dim*heads ==
               self.embed_size), "Embed_size need to be divisible by heads"
        '''
        The assert keyword is used to debug the code, assert keyword lets you test if a condition is True, if
        not the program will raise an AssertionError
        X="hello"
        assert x==="goodbye", "X should be 'hello'"
        '''
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads*self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        # Number of training examples(how many example are send at the same time) [Batch_size]
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        '''
        length are going to depend on where we use the attention mechanism and it is going to corresponse to
        the source and target sentence length. But since we don't know exactly where the mechanism is going 
        to be used i.e either in the encoder or which part of the decoder and these are going to vary so we 
        just use it abstractly (key length, query len, value length) but really they are going to correspond 
        to source length and target sentence length
        '''

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        '''
        Split embedding into self.heads pieces
        Note. (self.heads, self.head_dim), these two where we are spliting since before it was a single dimension (embed_size) and 
        now it is going to be (self.heads, self.head_dim)
        '''

        values = self.value(values)
        keys = self.key(keys)
        queries = self.query(queries)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads,     head_dim)
        # keys shape:    (N, key_len,   heads,     head_dim)
        # enery shape:   (N, head,      query_len, key_len)

        '''
        We can view, enery shape: (N, head, query_len, key_len). query_len as the target source sentence and the 
        key_len as the source sentence, its kind of say that for each word in our target how much should we pay
        attention to each word in our input in the source sentence
        '''
        '''
        print("Before Masking:\n Queries.shape:{}, Keys.shape:{}, Values.shape:{}, energy.shape:{}".format(
            queries.shape, keys.shape, values.shape, energy.shape
        ))
        '''
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        '''
        energy=energy.masked_fill(mask==0, float("-le20"))
        if mask==0 then fill with -le20 i.e a very small -ve no. instead of 0 to prevent underflow
        '''

        attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3)

        '''
        Attention(Q,K,V)= softmax(Q.K^T/sqrt(dim_key)).V, If we have those -infinity (small -ve no.) in the mask they are goinf to be 0
        attention=torch.softmax(energy/(self.embed_size**(1/2)), dim=3), dim=3 means that we are normalizing across the key len.
        Depending upon where we use the attention mechanism for e.g. say key_len is the source sentence and query_len is target sentence
        lenth then that would say how much we would want to make the attention scores normalized to 1 across the source sentences. E.g.
        if the first 0.8 means we are applying 80% attention to the first word in the source sentence
        '''
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim)  # concatenate hence reshape
        # attention shape: (N, head, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum(out shape: (N, query_len, heads, head_dim)) then flatten the last two dimension
        '''
        print("Atter Masking:\n Queries.shape:{}, Keys.shape:{}, Values.shape:{}, energy.shape:{}, out.shape:{}".format(
            queries.shape, keys.shape, values.shape, energy.shape, out.shape
        ))
        '''
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 dropout,
                 forward_expansion
                 ):
        super(TransformerBlock, self).__init__()

        self.selfattention = SelfAttention(embed_size, heads)
        # self.norm_1 = nn.LayerNorm(embed_size, elementwise_affine=False)
        # self.norm_2 = nn.LayerNorm(embed_size, elementwise_affine=False)
        # self.norm_1 = RMSNorm(embed_size)
        # self.norm_2 = RMSNorm(embed_size)
        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.selfattention(value, key, query, mask)
        # Add skip connection, run through normalization and finally dropout
        x = query + self.dropout(self.norm_1(attention))
        forward = self.feed_forward(x)
        out = x + self.dropout(self.norm_2(forward))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, device):
        super(PositionalEncoding, self).__init__()
        self.device = device

    def get_angles(self, pos, i, embed_size):  # pos: (seq_lenth,1) i:(1, embed_size)
        # pos: array of all the position of the inputs. Here integer from 0-19
        # i: array of all the possible dimension of the embedding
        # i/2 since we are going to store both val for even and odd hence 1/2 and 1/2
        angles = 1/np.power(10000., (2*(i//2))/np.float32(embed_size))
        # 2*(i/2): In our formula we apply sin to all the even so, i does not represent the actual dim
        # PE(pos, 2i) represent the dim
        # 2*(i/2) is just a conenient way to split the dim between the odd and even nos. Say, we want to compute the angle for dim=10 means
        # we could want to refer to PE(pos, 2i) and i=5. For dim=11 we would like to refer to PE(pos, 2i+1) and i=5
        # So angles will have each results twice one for sin and cosine
        return pos*angles  # Shape(seq_length, d_model)

    def forward(self, inputs):
        # as list since the default type is tf.tensor shape
        seq_length = inputs.shape[-2]
        embed_size = inputs.shape[-1]

        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],  # np.newaxis create an empty dim
                                 np.arange(embed_size)[np.newaxis, :],
                                 embed_size)

        # angle[:,0::2] means we want to access all the index starting from 0 to the end and the step will be 2 i.e even pos
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # pos_encoding=angles[np.newaxis, ...] #Since we want to work in batch
        pos_encoding = torch.from_numpy(angles).type(torch.float32)
        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.to(self.device)

        return inputs + pos_encoding


class EncoderContext(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device
    ):

        super(EncoderContext, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # Comment below line if we don't transforming embedding layer
        self.mapping = nn.Linear(embed_size * 2, embed_size)
        self.relu = nn.ReLU()

        #self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # self.norm = nn.LayerNorm(embed_size, elementwise_affine=False)
        # self.norm = RMSNorm(embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask, src_embed):
        out = self.position_embedding(self.word_embedding(x))
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        # print("out.shape:{}".format(out.shape))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        g = self.relu(self.mapping(torch.cat((out, src_embed), -1)))
        out = self.norm(torch.mul(
            (torch.ones(g.shape, dtype=torch.float32).to(self.device) - g), out) + torch.mul(g, src_embed))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # Comment below line if we don't transforming embedding layer
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        out = self.position_embedding(self.word_embedding(x))
        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        # print("out.shape:{}".format(out.shape))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_size,
                 heads,
                 dropout,
                 forward_expansion,
                 device
                 ):
        super(DecoderBlock, self).__init__()
        # self.norm = nn.LayerNorm(embed_size, elementwise_affine=False)
        # self.norm = RMSNorm(embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.selfattention = SelfAttention(embed_size=embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.selfattention(x, x, x, trg_mask)
        query = x + self.dropout(self.norm(attention))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(device)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout,
                             forward_expansion, device)
                for _ in range(num_layers)
            ]
        )
        self.embed_size = embed_size
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        out = self.position_embedding(self.word_embedding(x))
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(out)
        return out

import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer

# Class for input embedding
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        
# Class for Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        # Create a tensor of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        # Create a vector of shape (seq_len, 1)
        pos = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # We'll get 'batch_size' also so making the shape: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

# Class for Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.beta 

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        x = self.relu(self.linear2)
        
        return x 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.d_k = self.d_model // self.num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (bs, head, seq_len, dk) @ (bs, head, dk, seq_len) --> (bs, head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1) # (bs, head, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, head, d_k)---(transpose)--> (batch_size, head, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (bs, head, seq_len, d_k) --> (bs, seq_len, head, d_k) --> (bs, seq_len, d_model(h*d_k))
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.num_heads)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadAttention, ffn: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.attention_block = attention_block
        self.ffn = ffn 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): # src_mask is for padding so that their impact is zero
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ffn)

        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, ffn: FeedForwardNetwork, dropout):
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.ffn = ffn
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda: self.attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.ffn)

        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (bs, seq_len, d_model) --> (bs, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.encoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj(x)


def build_transformer(src_vocab: int, tgt_vocab: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Create IP/OP Embedding Layer
    src_embed = InputEmbedding(d_model, src_vocab)
    tgt_embed = InputEmbedding(d_model, tgt_vocab)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(attention_block, ffn, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        attention_block = MultiHeadAttention(d_model, h, dropout)
        cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(attention_block, cross_attention_block, ffn, dropout)
        decoder_blocks.append(decoder_block)

    # Create the Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection = ProjectionLayer(d_model, tgt_vocab)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    # Initialize the transformer params to make training faster
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


    






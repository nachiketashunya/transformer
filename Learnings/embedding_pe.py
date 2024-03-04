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
        pos = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model//2,)

        print("Div: ", div)
        print("Pos: ", pos)
        print("Pos * Div: ", pos * div)
        print("sin(pos*div): ", torch.sin(pos * div))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        
        print("PE[1]: ", pe[1])
        # We'll get 'batch_size' also so making the shape: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

d_model = 6 # Always take even number
vocab_size = 30522

# Define the tokenizer model (you can replace 'bert-base-uncased' with any other model you prefer)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example sentences
sentences = ['I am Nachiketa', 'I am very lazy', 'You are very hard working', "I'll regret in my life"]

# Tokenize the sentences
tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# print(tokens['input_ids'])
"""
Output - 
        tensor([[  101,  1045,  2572,  6583,  5428,  3489,  2696,   102,     0],
                [  101,  1045,  2572,  2200, 13971,   102,     0,     0,     0],
                [  101,  2017,  2024,  2200,  2524,  2551,   102,     0,     0],
                [  101,  1045,  1005,  2222,  9038,  1999,  2026,  2166,   102]])

"""

ie = InputEmbedding(d_model, vocab_size)
embedded_input = ie(tokens['input_ids'])

# print("Shape of embeeded input: ", embedded_input.shape)
# torch.Size([4, 9, 5])


# print("Embeeded input: ", embedded_input[0])
"""
So for batch = 1, I got (9, 5) matrix. 
It means for each token I got vector of shape (5, ). In this case (d_model,)

    [[-2.2261, -0.7671, -0.5595, -0.2330, -0.3341],
    [ 1.0317, -2.0473,  0.3365,  0.5312, -3.4365],
    [ 1.3434,  0.8677,  0.4315,  4.5054,  2.0255],
    [ 1.1296,  0.4766,  1.1545, -0.3509, -1.3532],
    [ 1.5247, -1.1562, -1.6797, -2.5917, -2.1248],
    [ 1.6725, -6.0158,  2.9235,  0.9017, -2.3144],
    [ 2.2851,  1.5793,  0.1727,  0.1979, -2.5676],
    [-0.0964,  0.4791, -2.9013, -3.9016,  0.5049],
    [-0.4271, -1.5615, -0.7327,  2.5981,  0.3222]]


Purpose of Input Embedding:

    Input embedding is created to represent words or tokens in a format suitable for neural network processing.
    It provides a dense vector representation for each word or token in the input sequence.

Dense Representation and Semantic Relationships:

    The dense vector representation captures semantic relationships between words.
    This allows the model to learn patterns and associations in the data, enhancing its understanding of language.

Reasons for Using Input Embeddings:
Dimensionality Reduction:

    Embeddings reduce the dimensionality of the input space.
    Instead of one-hot encoding with a vocabulary-sized vector, embeddings compress information into a fixed-size dense vector (e.g., 512 dimensions).
    This aids in managing input complexity and capturing meaningful patterns.

Semantic Information:

    Embeddings capture semantic information, placing words with similar meanings closer together in the embedding space.
    Crucial for tasks like natural language understanding, facilitating the model's comprehension of word relationships.

Learnable Parameters:

    Embeddings serve as learnable parameters in a neural network.
    During training, the model adjusts embedding vectors based on the specific task, allowing continuous improvement in word representations.

Generalization:

    Embeddings support the model in generalizing well to unseen words.
    Even when encountering unfamiliar words during training, the embedding space enables the model to make reasonable predictions based on learned relationships.

"""

pe = PositionalEncoding(d_model, embedded_input.size(1), dropout=0.3)
encoded = pe(embedded_input)

print(encoded[0])
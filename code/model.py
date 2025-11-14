import torch
import numpy as np
from torch.nn import Embedding, Linear
from torch.nn.parameter import Parameter
from typing import List


class PositionalEncoding(torch.nn.Module):

    """
    Implements the sinusoidal positional encoding as described in the "Attention is All You Need" paper.
    This module adds positional information to the input embeddings to help the model understand the
    order of tokens.
    
    Args:
        d_model (int): The dimension of the embeddings.
        max_length (int): The maximum length of the input sequences.
    """

    def __init__(self, d_model: int = 256, max_length : int = 64):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # position term
        pos = torch.arange(0, max_length).unsqueeze(1)

        ##10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        # init the positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        
        ##even terms in embedding dim
        pe[:, 0::2] = torch.sin(pos * div_term)

        ##odd terms in embedding dim
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_length, d_model)

        ## Non trainable, so register as buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encodings added
        """
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class LayerNorm(torch.nn.Module):

    """    
    Implementation of Layer Normalization as described in the "Layer Normalization" paper.
    
    Args:
        features (int): The number of features in the input tensor.
        eps (float): A small value to avoid division by zero during normalization.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(torch.ones(features))
        self.beta = Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, features)

        Returns:
            Tensor of the same shape as input with layer normalization applied
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadAttention(torch.nn.Module):

    """ 
    Implementation of Multi-Head Attention mechanism as described in the "Attention is All You Need" paper.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout rate to apply after attention.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None, past_layer_cache : tuple = None) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (batch_size, seq_len, d_model)
            key: Tensor of shape (batch_size, seq_len, d_model)
            value: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying multi-head attention
        """
        batch_size = query.size(0)

        ##The reshaping and transposing below enables parallel computation of attention across multiple heads

        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if past_layer_cache is not None:

            past_k, past_v = past_layer_cache

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_layer_cache = (k,v)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        self.last_attn = attn_weights  # Store attention weights for visualization

        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.linear_out(attn_output)
        return output, present_layer_cache

class ResidualConnection(torch.nn.Module):

    """
    Implements a residual connection followed by layer normalization.
    
    Args:
        size (int): The number of features in the input tensor.
        dropout (float): Dropout rate to apply after the residual connection.
    """

    def __init__(self, size: int, dropout: float = 0.2):
        super(ResidualConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: torch.nn.Module) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, features)
            sublayer: A sublayer module to apply to the input

        Returns:
            Tensor of the same shape as input after applying residual connection and layer normalization
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class FFN(torch.nn.Module):

    """
    Implements the Position-wise Feed-Forward Network as described in the "Attention is All You Need" paper.
    
    Args:
        d_model (int): The dimension of the input embeddings.
        d_ff (int): The dimension of the feed-forward layer.
        dropout (float): Dropout rate to apply after the feed-forward layer.
    """

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super(FFN, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying feed-forward network
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class DecoderBlock(torch.nn.Module):

    """    
    Implementation of a Transformer Decoder Block as described in the "Attention is All You Need" paper.
    
    Args:
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward network.
        dropout (float): Dropout rate to apply in various parts of the block.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        super(DecoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)

        self.feed_forward = torch.nn.Sequential(
            Linear(d_model, d_ff),
            torch.nn.ReLU(),
            Linear(d_ff, d_model)
        )
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                past_layer_cache: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying the decoder block
        """

        norm_x = self.residual1.layer_norm(x)
        attn_output, present_layer_cache = self.self_attn(
            norm_x, norm_x, norm_x, mask, past_layer_cache)
        
        x = x + self.residual1.dropout(attn_output)

        x = self.residual2(x, self.feed_forward)
        
        return x, present_layer_cache

class DecoderTransformer(torch.nn.Module):

    """
    Implementation of a Transformer Decoder as described in the "Attention is All You Need" paper.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward network.
        num_layers (int): The number of decoder layers.
        max_length (int): The maximum length of the input sequences.
        dropout (float): Dropout rate to apply in various parts of the model.
        pretrained_embeddings (Embedding): Optional pre-trained embedding layer.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8, d_ff: int = 512, num_layers: int = 3, max_length: int = 64, dropout: float = 0.1, pretrained_embeddings: Embedding = None):
        super(DecoderTransformer, self).__init__()

        # Use pretrained embeddings if provided, otherwise create new ones
        if pretrained_embeddings is not None:
            self.embedding = pretrained_embeddings
            # If pretrained embeddings have different dimension, add a projection layer
            if pretrained_embeddings.embedding_dim != d_model:
                self.embedding_projection = Linear(pretrained_embeddings.embedding_dim, d_model)
            else:
                self.embedding_projection = None
        else:
            self.embedding = Embedding(vocab_size, d_model)
            self.embedding_projection = None
            
        self.positional_encoding = PositionalEncoding(d_model, max_length)

        self.layers = torch.nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.layer_norm = LayerNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                past_kv_cache: List = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            mask: Optional tensor for masking (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) after applying the decoder transformer
        """

        batch_size, seq_len = x.size()

        if past_kv_cache is None:
        # Create causal mask if not provided
            if mask is None:
                # Create mask with shape (1, 1, seq_len, seq_len) for broadcasting
                mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=x.device), diagonal=1).bool()
        else:
            mask = None
        

        x = self.embedding(x)
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)

            
        x = x * torch.sqrt(torch.tensor(self.positional_encoding.d_model, dtype=torch.float32))

        if past_kv_cache is not None:
            past_seq_len = past_kv_cache[0][0].size(2)

            pos_encoding = self.positional_encoding.pe[:, past_seq_len:past_seq_len + seq_len, :]
        
        else:
            pos_encoding = self.positional_encoding.pe[:, :seq_len, :]

        x = x + pos_encoding

        present_kv_cache = []
        
        for i, layer in enumerate(self.layers):
            
            layer_cache = past_kv_cache[i] if past_kv_cache is not None else None

            x, new_layer_cache = layer(x, mask, layer_cache)
            present_kv_cache.append(new_layer_cache)

        x = self.layer_norm(x)
        output = self.output_linear(x)

        return output, present_kv_cache

from datasets import load_dataset, concatenate_datasets, load_from_disk, DownloadMode
from torch.utils.data import DataLoader
from torch.nn import Linear, Embedding, Parameter
import torch.optim as optim
import torch
import numpy as np
import random
import os
from tqdm.auto import tqdm
import spacy
import re
from collections import Counter
from torchtext.vocab import vocab, FastText
from torchtext.data.metrics import bleu_score
from collections import OrderedDict
from typing import List, Dict
import argparse
import wandb


##For reporducibility
def set_seed(seed = 5758):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

cache_location = '/home/anwesh/scratch/hf_cache/'
dataset = load_dataset('roneneldan/TinyStories', cache_dir=cache_location)

os.environ['SPACY_DATA'] = '/home/anwesh/scratch/spacy_data/'

full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

# _spacy_model = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

# def spacy_tokenize(text):
#     """Tokenize a single string using spaCy."""
#     return [token.text.lower() for token in _spacy_model(text) if not token.is_space]

# def batch_spacy_tokenize(batch):
#     """Tokenize a batch of samples using spaCy's nlp.pipe."""
#     texts = batch['text']
#     return {'tokens': [[token.text.lower() for token in doc if not token.is_space] for doc in _spacy_model.pipe(texts, batch_size=512*4, n_process=1)]}


tokenized_full = load_from_disk("/home/anwesh/scratch/ELL8299 Project/tokenized_tiny_stories_data")
print("Tokenized dataset loaded from disk.")

train_size = len(dataset['train']) 
valid_size = len(dataset['validation'])  

tokenized_train = tokenized_full.select(range(0, train_size))
tokenized_valid = tokenized_full.select(range(train_size, train_size + valid_size))

tiny_stories_vocab = torch.load("/home/anwesh/scratch/ELL8299 Project/tiny_stories_vocab.pt")
tiny_stories_vocab.set_default_index(tiny_stories_vocab['<pad>'])
fasttext_vectors = FastText(language='en', cache='/home/anwesh/scratch/ELL8299 Project/vector_cache')

VOCAB_SIZE = len(tiny_stories_vocab)        
EMBEDDING_DIM = 300

def create_embedding_layer(vocab_size = VOCAB_SIZE, 
                           embedding_dim = EMBEDDING_DIM) -> Embedding:
    """Create an embedding layer initialized with FastText embeddings."""


    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Initialize with small random values
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))

    for idx, token in enumerate(tiny_stories_vocab.get_itos()):
        try:
            # Get the vector for the token from FastText
            vector = fasttext_vectors.get_vecs_by_tokens(token)
            embedding_matrix[idx] = vector.cpu().numpy()
        except KeyError:
            pass

    pad_idx = tiny_stories_vocab['<pad>']
    embedding_matrix[pad_idx] = np.zeros(embedding_dim)

    embedding_layer = Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float32),  # Explicitly set to float32
        freeze=True     # Set to False if you want to fine-tune the embeddings
    )

    return embedding_layer

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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
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
        return output

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional tensor for masking (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying the decoder block
        """
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        return x

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            mask: Optional tensor for masking (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) after applying the decoder transformer
        """

        batch_size, seq_len = x.size()

        # Create causal mask if not provided
        if mask is None:
            # Create mask with shape (1, 1, seq_len, seq_len) for broadcasting
            mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=x.device), diagonal=1).bool()

        x = self.embedding(x)
        
        # Project embeddings if dimensions don't match
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)
            
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)
        output = self.output_linear(x)
        return output

def create_decoder_transformer(seq_len: int, device: str, pretrained_embeddings: Embedding, num_heads: int = 8, num_layers = 3) -> DecoderTransformer:

    model = DecoderTransformer(
        vocab_size=len(tiny_stories_vocab),
        d_model=256,
        num_heads=num_heads,
        d_ff=512,
        num_layers=num_layers,
        max_length=seq_len,
        dropout=0.1,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    return model

# seq_len = 64
# embedding_layer = create_embedding_layer()


# decoder_model = create_decoder_transformer(
#     seq_len=seq_len, 
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#     pretrained_embeddings=embedding_layer
# )

def collate_fn(batch: List[Dict], vocab: vocab, seq_len: int) -> Dict[str, torch.Tensor]:
    
    """
    Custom collate function to prepare batches of data.
    """
    
    input_ids = []
    target_ids = []

    for sample in batch:
        tokens = sample['tokens']
        token_ids = [vocab['<sos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
        
        # Truncate or pad sequences to seq_len
        if len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        else:
            token_ids += [vocab['<pad>']] * (seq_len - len(token_ids))
        
        input_ids.append(token_ids[:-1])  # Input sequence
        target_ids.append(token_ids[1:])  # Target sequence (shifted by 1)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'target_ids': torch.tensor(target_ids, dtype=torch.long)
    }

def create_dataloaders(train_dataset, valid_dataset, vocab, seq_len, batch_size):

    if train_dataset is not None:

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, vocab, seq_len)
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab, seq_len)
    )

    return train_dataloader, valid_dataloader
    
# train_loader, valid_loader = create_dataloaders(tokenized_train, tokenized_valid, tiny_stories_vocab, seq_len, batch_size=128)    

def _ids_to_token_lists(batch_ids):
    """
    Convert a tensor of token ids (batch_size, seq_len) to a list of token lists,
    trimming at <eos> or <pad> and removing <sos>.
    """
    itos = tiny_stories_vocab.get_itos()
    pad_idx = tiny_stories_vocab['<pad>']
    sos_idx = tiny_stories_vocab['<sos>']
    eos_idx = tiny_stories_vocab['<eos>']

    token_lists = []
    for seq in batch_ids.cpu().numpy():
        toks = []
        for idx in seq:
            if idx == pad_idx:
                break
            if idx == sos_idx:
                continue
            if idx == eos_idx:
                break
            toks.append(itos[int(idx)])
        token_lists.append(toks)
    return token_lists


def train_decoder_transformer(train_loader, validation_loader, decoder_transformer,
                            lr=3e-4, weight_decay=0, num_epochs=10):
    """
    Train the decoder transformer model with early stopping and checkpointing.
    Tracks and logs BLEU for train and validation to wandb.
    """
    # Ensure wandb run exists
    wandb_dir = "/home/anwesh/scratch/ELL8299 Project/wandb"
    checkpoint_root = "/home/anwesh/scratch/ELL8299 Project/final_run_ckpts"
    
    run_name = wandb.run.name if (wandb.run is not None and wandb.run.name is not None) else "run"
    
    checkpoint_dir = os.path.join(checkpoint_root, run_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    # os.makedirs(wandb_dir, exist_ok=True)

    if os.path.exists(best_ckpt_path):
        print(f"Found existing checkpoint at {best_ckpt_path}, loading...")
        checkpoint = torch.load(best_ckpt_path, map_location=next(decoder_transformer.parameters()).device)
        wandb_run_id = checkpoint.get('wandb_run_id', None)


        if wandb_run_id and wandb.run is None:
            print(f"Resuming wandb run with ID: {wandb_run_id}")
            wandb.init(project="decoder-transformer", dir=wandb_dir, id=wandb_run_id, resume="must", reinit=True)
    
    if wandb.run is None:
        wandb.init(project="decoder-transformer", dir=wandb_dir, reinit=True)
        print(f"Started new wandb run with ID: {wandb.run.id}")
    
    # Create checkpoint directory using wandb run name
    
    optimizer = optim.Adam(decoder_transformer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tiny_stories_vocab['<pad>'])
    
    device = next(decoder_transformer.parameters()).device

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state_dict = None
    best_epoch = -1
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(best_ckpt_path):
        print(f"Found existing checkpoint at {best_ckpt_path}, loading...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)


        decoder_transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        best_epoch = checkpoint.get('epoch', -1)
        print(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    else:
        print("No existing checkpoint found, starting training from scratch.")
        start_epoch = 0

    for epoch in range(num_epochs):
        # Training phase
        decoder_transformer.train()
        total_train_loss = 0

        train_candidates = []
        train_references = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            target_seq = batch['target_ids'].to(device)  # shape (batch, seq_len)

            optimizer.zero_grad()
            output = decoder_transformer(input_ids)  # (batch, seq_len, vocab_size)

            # Predictions for BLEU (before flattening)
            predicted_ids = torch.argmax(output, dim=-1)  # (batch, seq_len)

            # Convert predicted and reference ids to token lists and accumulate
            preds_tokens = _ids_to_token_lists(predicted_ids)
            refs_tokens = _ids_to_token_lists(target_seq)
            # torchtext.bleu_score expects references as list of list(s) per hypothesis
            train_candidates.extend(preds_tokens)
            train_references.extend([[r] for r in refs_tokens])

            # Compute loss (flatten)
            output_flat = output.view(-1, output.size(-1))
            target_flat = target_seq.view(-1)

            loss = criterion(output_flat, target_flat)
            loss.backward()
            
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(decoder_transformer.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_perplexity = float(np.exp(avg_train_loss))

        
        # Compute training BLEU on first 10% of examples (to save time)
        try:
            total_examples = len(train_candidates)
            if total_examples == 0:
                train_bleu = 0.0
            else:
                k = max(1, int(0.1 * total_examples))
                train_candidates_subset = train_candidates[:k]
                train_references_subset = train_references[:k]
                train_bleu = float(bleu_score(train_candidates_subset, train_references_subset))
        except Exception as e:
            print(f"Warning: could not compute train BLEU: {e}")
            train_bleu = 0.0
        
        # Validation phase
        decoder_transformer.eval()
        total_val_loss = 0

        val_candidates = []
        val_references = []
        
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                target_seq = batch['target_ids'].to(device)

                output = decoder_transformer(input_ids)
                predicted_ids = torch.argmax(output, dim=-1)

                preds_tokens = _ids_to_token_lists(predicted_ids)
                refs_tokens = _ids_to_token_lists(target_seq)
                val_candidates.extend(preds_tokens)
                val_references.extend([[r] for r in refs_tokens])

                output_flat = output.view(-1, output.size(-1))
                target_flat = target_seq.view(-1)

                loss = criterion(output_flat, target_flat)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        val_perplexity = float(np.exp(avg_val_loss))

        # Compute validation BLEU
        try:
            total_examples = len(val_candidates)
            if total_examples == 0:
                val_bleu = 0.0
            else:
                k = max(1, int(0.1 * total_examples))
                val_candidates_subset = val_candidates[:k]
                val_references_subset = val_references[:k]
                val_bleu = float(bleu_score(val_candidates_subset, val_references_subset))
        except Exception as e:
            print(f"Warning: could not compute val BLEU: {e}")
            val_bleu = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Train BLEU: {train_bleu:.4f} | Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}, Val BLEU: {val_bleu:.4f}")

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'train_bleu': train_bleu,
            'val_bleu': val_bleu,
            'patience:': epochs_no_improve})

        early_stopping_delta = 0.01
        # Early stopping & checkpointing (improvement = strictly lower val loss)
        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            # Save best state (clone to CPU to avoid GPU memory issues)
            best_state_dict = {k: v.cpu().clone() for k, v in decoder_transformer.state_dict().items()}
            epochs_no_improve = 0
            best_epoch = epoch + 1

            # Save checkpoint dict to disk
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': best_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': dict(wandb.config) if wandb.config is not None else {},
                'wandb_run_id': wandb.run.id}
            torch.save(checkpoint, best_ckpt_path)
            print(f"Checkpoint saved to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Early stopping at epoch {epoch+1}.")
            break

        # At the end of the epoch print and log an input, target and output sample in text format
        sample_batch = next(iter(validation_loader))
        input_ids = sample_batch['input_ids'].to(device)
        target_seq = sample_batch['target_ids'].to(device)
        output = decoder_transformer(input_ids)
        predicted_ids = torch.argmax(output, dim=-1)

        # Prepare and log a single sample (first sample in batch)
        input_text = ' '.join([tiny_stories_vocab.get_itos()[idx] for idx in input_ids[0].cpu().numpy() if idx != tiny_stories_vocab['<pad>']])
        target_text = ' '.join([tiny_stories_vocab.get_itos()[idx] for idx in target_seq[0].cpu().numpy() if idx != tiny_stories_vocab['<pad>']])
        predicted_text = ' '.join([tiny_stories_vocab.get_itos()[idx] for idx in predicted_ids[0].cpu().numpy() if idx != tiny_stories_vocab['<pad>']])

        print(f"\nSample 1:\nInput: {input_text}\nTarget: {target_text}\nPredicted: {predicted_text}\n")

        wandb.log({
            'sample/input': input_text,
            'sample/target': target_text,
            'sample/predicted': predicted_text})

    # Restore best model state from disk if available
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        decoder_transformer.load_state_dict(ckpt['model_state_dict'])
        decoder_transformer.to(device)
        print(f"Model restored to best validation state from epoch {ckpt.get('epoch', best_epoch)} with Val Loss: {ckpt.get('val_loss', best_val_loss):.4f}")
    elif best_state_dict is not None:
        decoder_transformer.load_state_dict(best_state_dict)
        decoder_transformer.to(device)
        print(f"Model restored to best validation state from epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint available to restore.")


def eval_decoder_transformer(test_loader, decoder_transformer):
    """
    Evaluate the decoder transformer model and compute BLEU on the provided loader.
    """
    # Ensure wandb run exists for logging
    wandb_dir = "/home/anwesh/scratch/ELL8299 Project/wandb"
    os.makedirs(wandb_dir, exist_ok=True)
    if wandb.run is None:
        wandb.init(project="decoder-transformer", dir=wandb_dir, reinit=True)

    decoder_transformer.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tiny_stories_vocab['<pad>'])
    device = next(decoder_transformer.parameters()).device
    
    total_loss = 0

    candidates = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            target_seq = batch['target_ids'].to(device)

            output = decoder_transformer(input_ids)
            predicted_ids = torch.argmax(output, dim=-1)

            preds_tokens = _ids_to_token_lists(predicted_ids)
            refs_tokens = _ids_to_token_lists(target_seq)
            candidates.extend(preds_tokens)
            references.extend([[r] for r in refs_tokens])

            output_flat = output.view(-1, output.size(-1))
            target_flat = target_seq.view(-1)

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_perplexity_score = float(np.exp(avg_loss))
    print(f"Test Loss: {avg_loss:.4f}, Test Perplexity: {avg_perplexity_score:.4f}")
    
    try:
        test_bleu = float(bleu_score(candidates, references))
    except Exception as e:
        print(f"Warning: could not compute test BLEU: {e}")
        test_bleu = 0.0

    print(f"Test BLEU: {test_bleu:.4f}")
    
    # Log evaluation metrics
    wandb.log({
        'test_loss': avg_loss,
        'test_perplexity': avg_perplexity_score,
        'test_bleu': test_bleu})
    
    return avg_loss, avg_perplexity_score, test_bleu


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Decoder Transformer model.")
    parser.add_argument('--seq_len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--wandb_project', type=str, default='decoder-transformer', help='WandB project name')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training and evaluation')

    args = parser.parse_args()

    set_seed()

    # cache_location = '/home/anwesh/scratch/hf_cache/'
    # dataset = load_dataset('roneneldan/TinyStories', cache_dir=cache_location)

    # os.environ['SPACY_DATA'] = '/home/anwesh/scratch/spacy_data/'

    # full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

    wandb_run_id = None
    run_name = f"seq{args.seq_len}_layers{args.num_layers}_heads{args.num_heads}_lr{args.lr}_wd{args.weight_decay}"
    checkpoint_path = f"/home/anwesh/scratch/ELL8299 Project/final_run_ckpts/{run_name}/best_model.pt"
    wandb_dir = "/home/anwesh/scratch/ELL8299 Project/wandb"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
        wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    config={
        'seq_len': args.seq_len,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs}
    
    if wandb_run_id:
        wandb.init(project=args.wandb_project,
                   id = wandb_run_id,
                   resume="allow",
                   name = run_name,
                   config=config,
                   dir = wandb_dir)
        
    else:
        wandb.init(project=args.wandb_project,
                   name = run_name,
                   config=config,
                   dir = wandb_dir)


    print("Starting training with the following configuration:")
    
    for key, value in wandb.config.items():
        print(f"{key}: {value}")

    seq_len = args.seq_len
    
    embedding_layer = create_embedding_layer()

    decoder_model = create_decoder_transformer(
        seq_len=seq_len, 
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        device=args.device,
        pretrained_embeddings=embedding_layer)

    train_loader, valid_loader = create_dataloaders(tokenized_train, tokenized_valid, 
                                                    tiny_stories_vocab, seq_len, batch_size=args.batch_size)
    
    train_decoder_transformer(
        train_loader, 
        valid_loader, 
        decoder_transformer=decoder_model, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs)

    eval_decoder_transformer(
        decoder_transformer=decoder_model, 
        test_loader=valid_loader)

    wandb.finish()
    

if __name__ == "__main__":
    main()
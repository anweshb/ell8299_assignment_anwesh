from datasets import load_dataset, concatenate_datasets
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
from collections import OrderedDict
from typing import List, Dict

def set_seed(seed = 5758):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# load spaCy model once
_spacy_model = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

def spacy_tokenize(text):
    """Tokenize a single string using spaCy."""
    return [token.text.lower() for token in _spacy_model(text) if not token.is_space]

def batch_spacy_tokenize(batch):
    """Tokenize a batch of samples using spaCy's nlp.pipe."""
    texts = batch['text']
    return {'tokens': [[token.text.lower() for token in doc if not token.is_space] for doc in _spacy_model.pipe(texts, batch_size=512*4, n_process=1)]}


if __name__ == "__main__":
    
    set_seed()
    print("Starting tokenization and vocabulary building...")

    cache_location = '/home/anwesh/scratch/hf_cache/'
    dataset = load_dataset('roneneldan/TinyStories', cache_dir=cache_location)

    os.environ['SPACY_DATA'] = '/home/anwesh/scratch/spacy_data/'


    full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

    print("Full dataset size:", len(full_dataset))

    _spacy_model = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    # 2. Tokenize all samples for vocab creation
    tokenized_full = full_dataset.map(batch_spacy_tokenize, batched=True, batch_size=512*4, remove_columns=['text'], num_proc=8)
    print("Tokenization complete.")
    tokenized_full.save_to_disk("/home/anwesh/scratch/ELL8299 Project/tokenized_tiny_stories_data")
    print("Tokenized dataset saved.")

    # 3. Build vocabulary from all tokens
    all_tokens = [token for tokens in tokenized_full['tokens'] for token in tokens]
    special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
    vocab_counter = Counter(all_tokens)
    tiny_stories_vocab = vocab(vocab_counter, specials=special_tokens, min_freq=5, special_first=True)
    tiny_stories_vocab.set_default_index(tiny_stories_vocab['<unk>'])

    print("Vocabulary built. Saving vocabulary object...")
    torch.save(tiny_stories_vocab, "/home/anwesh/scratch/ELL8299 Project/tiny_stories_vocab.pt")

    print("Vocabulary saved.")


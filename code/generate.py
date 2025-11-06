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
from train import create_decoder_transformer, create_embedding_layer, create_dataloaders
from tokenization import spacy_tokenize


tiny_stories_vocab = torch.load("/home/anwesh/scratch/ELL8299 Project/tiny_stories_vocab.pt")

def set_seed(seed = 5758):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# def instantiate_model(model_class, 
#                      checkpoint_path: str, device: torch.device):
    
#     """Instantiates the model and loads the weights from the checkpoint."""

#     model = model_class(
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()

#     return model


def prepare_prompt(input_text: str, vocab = tiny_stories_vocab):
    

    """Prepares the input prompt for generation by tokenizing, converting to IDs,
    and truncating if necessary."""

    tokenized_text = spacy_tokenize(input_text) if isinstance(input_text, str) else input_text

    sos = vocab['sos']
    input_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokenized_text]
    input_ids = [sos] + input_ids
 
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # shape: (1, seq_len)
    
    return input_tensor

def generate(input_prompt: str, model,
             device: torch.device, vocab, 
             max_output_length: int = 100,
             temperature = 0.2, top_k = 50, 
             stochastic = True) -> str:
    
    input_tensor = prepare_prompt(input_prompt).to(device)
    
    model.to(device)
    model.eval()

    generated_tokens = []

    with torch.no_grad():
        
        for _ in range(max_output_length):

            if input_tensor.size(1) > model.positional_encoding.pe.size(1):
                # If input length exceeds model's max positional encoding length, break
                break

            #Get model predictions    
            logits = model(input_tensor)        # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :] / (temperature)  # (1, vocab_size)

            if top_k:
                values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                kth = values[:, -1].unsqueeze(-1)
                mask = next_token_logits < kth
                next_logits = next_token_logits.masked_fill(mask, -float('Inf'))
            
            probs = torch.softmax(next_token_logits, dim=-1)  # (1, vocab_size)

            if stochastic:
                next_id = int(torch.multinomial(probs, num_samples=1))
            else:
                next_id = int(torch.argmax(probs, dim=-1))
            
            if next_id == vocab['eos']:
                break

            generated_tokens.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=device) #  -> (1, 1) shape
            input_ids = torch.cat([input_tensor, next_tensor], dim=-1)  # Append to input

    return ' '.join([vocab.get_itos()[token_id] for token_id in generated_tokens])


def path_to_params(checkpoint_path: str):
    """Extracts model parameters from the checkpoint path."""
    checkpoint_name = os.path.basename(checkpoint_path)
    params = {}
    for part in checkpoint_name.split('_'):
        if part.startswith('seq'):
            params['seq_len'] = int(part[3:])
        elif part.startswith('layers'):
            params['num_layers'] = int(part[6:])
        elif part.startswith('heads'):
            params['num_heads'] = int(part[5:])
    return params



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default = '/home/anwesh/scratch/ELL8299 Project/model_ckpts/seq64_layers3_heads8_lr0.0003_wd0.0/best_model.pt',
                        help='Path to the model checkpoint.')
    parser.add_argument('--input_prompt', type=str,
                        default = "Write a story about a forest.",
                        help='Input prompt for text generation.')
    parser.add_argument('--max_output_length', type=int, default=50,
                        help='Maximum length of the generated output.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k filtering for sampling.')
    parser.add_argument('--stochastic', default=True, 
                        help='Use stochastic sampling if set; otherwise use greedy decoding.')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on.') 
    args = parser.parse_args()
    
    print("INITING MODEL...")

    ##Get params like seq_len, num_heads and num_layers from checkpoint name
    params = path_to_params(args.checkpoint_path)

    embedding_layer = create_embedding_layer(vocab_size=len(tiny_stories_vocab), 
                                             embedding_dim=300)
        
    model = create_decoder_transformer(
        seq_len=params['seq_len'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'], 
        device=args.device,
        pretrained_embeddings=embedding_layer)
    
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))
    # model.to(args.device)
    # model.eval()

    print("MODEL INITED.")

    print()
    print()
    print()

    print("GENERATING TEXT...")

    generated_text = generate(args.input_prompt, model, args.device, tiny_stories_vocab,
                              args.max_output_length, args.temperature,
                              args.top_k, args.stochastic)

    print("Generated Text:\n", generated_text)


if __name__ == "__main__":
    main()
            

            


    







    






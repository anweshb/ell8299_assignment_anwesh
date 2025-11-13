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
from train import create_decoder_transformer, create_embedding_layer, cache_location, tokenized_valid, collate_fn
from tokenization import spacy_tokenize
import math

tiny_stories_vocab = torch.load("/home/anwesh/scratch/ELL8299 Project/tiny_stories_vocab.pt")


def prepare_prompt(input_text: str, max_length, vocab = tiny_stories_vocab):
    

    """Prepares the input prompt for generation by tokenizing, converting to IDs,
    and truncating if necessary."""
    input_text = input_text.lower()
    tokenized_text = spacy_tokenize(input_text) if isinstance(input_text, str) else input_text

    sos = vocab['<sos>']
    input_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokenized_text]

    max_tokens = max_length - 1  # accounting for <sos> token

    if len(input_ids) > max_tokens:
        input_ids = input_ids[-max_tokens:]  # Truncate from the beginning
    
    input_ids = [sos] + input_ids 
    input_tensor = torch.tensor(input_ids).unsqueeze(0)  # shape: (1, seq_len)
    
    return input_tensor

def generate(input_prompt: str,
             device: str,
             model, vocab, max_length,
             max_output_length: int = 100,
             temperature = 0.5, top_k = 20, 
             stochastic = True,
             reference_text = None) -> dict:
    
    input_tensor = prepare_prompt(input_prompt, max_length, vocab)
    input_tensor = input_tensor.to(torch.device(device))

    model.to(device)
    model.eval()

    generated_tokens = []
    token_log_probs = []

    with torch.no_grad():
        
        for _ in range(max_output_length):

            if input_tensor.size(1) >= model.positional_encoding.pe.size(1):
                #if input length exceeds model's max positional encoding length, break
                break

            #get model predictions    
            logits = model(input_tensor)        # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :] / (temperature)  # (1, vocab_size)

            if top_k:
                values, indices = torch.topk(next_token_logits, top_k, dim=-1)
                kth = values[:, -1].unsqueeze(-1)
                mask = next_token_logits < kth
                next_token_logits = next_token_logits.masked_fill(mask, -float('Inf'))
            
            probs = torch.softmax(next_token_logits, dim=-1)  # (1, vocab_size)

            if stochastic:
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(probs, dim=-1).item())
            
            if next_id == vocab['<eos>']:
                break

            generated_tokens.append(vocab.get_itos()[next_id])
            next_tensor = torch.tensor([[next_id]], device=torch.device(device)) #  -> (1, 1) shape
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)  # Append to input

            token_prob = probs[0, next_id].item()
            token_log_prob = torch.log(probs[0, next_id]).item()
            token_log_probs.append(token_log_prob)

    if len(token_log_probs) > 0:

        perplexity_per_token = [math.exp(-log_prob) for log_prob in token_log_probs]
        mean_perplexity = sum(perplexity_per_token)/len(perplexity_per_token)

        if reference_text:

            reference_text = ' '.join(reference_text).lower()

            reference_tokens = [spacy_tokenize(reference_text.lower())]
            references = [reference_tokens]

            bleu = bleu_score([generated_tokens], references)
        
        else: 
            bleu = None

    generated_text = ' '.join(generated_tokens)
    return generated_text, mean_perplexity, bleu


def generate_from_tinystories(args, model, params, tokenized_validation = tokenized_valid):
    
   #Sample 50 random examples from validation set
   trimmed_tokenized_valid = np.random.choice(tokenized_validation, 50)

   ground_truth_completions = [example['tokens'][5:] for example in trimmed_tokenized_valid]
   tiny_stories_prompts = [example['tokens'][:5] for example in trimmed_tokenized_valid]
   perplexity_per_token_list = []
   bleu_list = []
   generated_completions = []

   for prompts, ground_truth in zip(tiny_stories_prompts, ground_truth_completions):
        prompt_text = ' '.join(prompts)
        generated_text, perplexity_per_token , bleu = generate(input_prompt= prompt_text, model = model, device = args.device, 
                                                            vocab = tiny_stories_vocab, max_output_length = args.max_output_length, 
                                                            max_length = params['seq_len'], temperature = args.temperature,
                                                            top_k = args.top_k, stochastic = args.stochastic, reference_text = ground_truth)
        
        generated_completions.append(generated_text.split())
        perplexity_per_token_list.append(perplexity_per_token)
        bleu_list.append(bleu)
    

   mean_perplexity = sum(perplexity_per_token_list)/len(perplexity_per_token_list)
   mean_bleu = sum(bleu_list)/len(bleu_list)

   print("Average Perplexity per token on 50 samples from TinyStories validation set: ", mean_perplexity)
   print("Average BLEU score on 50 samples from TinyStories validation set: ", mean_bleu)


   return (tiny_stories_prompts, generated_completions), perplexity_per_token_list, bleu_list
    
        
    


def path_to_params(checkpoint_path: str):
    """Extracts model parameters from the checkpoint path."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dir_name = os.path.basename(checkpoint_dir)
    
    params = {}
    for part in dir_name.split('_'):
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
                        default = '/home/anwesh/scratch/ELL8299 Project/final_run_ckpts/seq128_layers3_heads16_lr0.0003_wd0.0/best_model.pt',
                        help='Path to the model checkpoint.')
    parser.add_argument('--custom_prompt', type=bool,
                        default = False)
    parser.add_argument('--input_prompt', type=str, 
                        default = "Write a story about engineers.",
                        help='User input prompt for text generation.')
    parser.add_argument('--max_output_length', type=int, default=64,
                        help='Maximum length of the generated output.')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-k filtering for sampling.')
    parser.add_argument('--stochastic', type=bool, default=True, 
                        help='Use stochastic sampling if set; otherwise use greedy decoding.')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on.') 
    args = parser.parse_args()
    
    print("INITING MODEL...")

    ##Get params like seq_len, num_heads and num_layers from checkpoint name
    params = path_to_params(args.checkpoint_path)
    # print("PRINTING PARAMS FROM CHECKPOINT PATH:")
    # print(params)

    embedding_layer = create_embedding_layer(vocab_size=len(tiny_stories_vocab), 
                                             embedding_dim=300)

    print("MODEL CREATED. \n\n")
    model = create_decoder_transformer(
    seq_len=params['seq_len'],
    num_heads=params['num_heads'],
    num_layers=params['num_layers'], 
    device=args.device,
    pretrained_embeddings=embedding_layer)

    print("LOADING CHECKPOINT... \n\n")
    ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    # model.to(args.device)
    # model.eval()

    if args.custom_prompt:
        user_input_prompt = input("Welcome to Anwesh's Language Model! Enter your prompt:")
    

        print("GENERATING TEXT...")

        bleu = None

        generated_text, _, _ = generate(input_prompt = args.input_prompt, model = model, device = args.device, 
                                                          vocab = tiny_stories_vocab, max_output_length = args.max_output_length, 
                                                          max_length = params['seq_len'], temperature = args.temperature,
                                                          top_k = args.top_k, stochastic = args.stochastic)

        print("Generated Text:\n", generated_text)
        
    else:
        print("GENERATING TEXT FROM TINYSTORIES VALIDATION SET...")
    
        (trimmed_prompts, generated_texts), perplexity_list, bleu_list = generate_from_tinystories(args, model, params, tokenized_validation = tokenized_valid)


        for items in zip(trimmed_prompts, generated_texts, perplexity_list, bleu_list):
            prompt = ' '.join(items[0])
            generated = ' '.join(items[1])
            perplexity = items[2]
            bleu = items[3]

            print(f"Prompt: {prompt}\nGenerated Text: {generated}\nPerplexity per token: {perplexity}\nBLEU Score: {bleu}\n\n") 


if __name__ == "__main__":
    main()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default = '/home/anwesh/scratch/ELL8299 Project/final_run_ckpts/seq128_layers3_heads16_lr0.0003_wd0.0/best_model.pt',
    #                     help='Path to the model checkpoint.')
    # parser.add_argument('--input_prompt', type=str, 
    #                     default = "Write a story about engineers.",
    #                     help='Input prompt for text generation.')
    # parser.add_argument('--max_output_length', type=int, default=64,
    #                     help='Maximum length of the generated output.')
    # parser.add_argument('--temperature', type=float, default=0.2,
    #                     help='Temperature for sampling.')
    # parser.add_argument('--top_k', type=int, default=10,
    #                     help='Top-k filtering for sampling.')
    # parser.add_argument('--stochastic', type=bool, default=True, 
    #                     help='Use stochastic sampling if set; otherwise use greedy decoding.')
    # parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu',
    #                     help='Device to run the model on.') 
    # args = parser.parse_args()
    
    # print("INITING MODEL...")

    # ##Get params like seq_len, num_heads and num_layers from checkpoint name
    # params = path_to_params(args.checkpoint_path)

    # # print("PRINTING PARAMS FROM CHECKPOINT PATH:")
    # # print(params)

    # embedding_layer = create_embedding_layer(vocab_size=len(tiny_stories_vocab), 
    #                                          embedding_dim=300)

    # model = create_decoder_transformer(
    # seq_len=params['seq_len'],
    # num_heads=params['num_heads'],
    # num_layers=params['num_layers'], 
    # device=args.device,
    # pretrained_embeddings=embedding_layer)


    # ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    # model.load_state_dict(ckpt['model_state_dict'])

    # returned_tokenized_valid = generate_from_tinystories(args, model, params, tokenized_validation = tokenized_valid)

    # for sentences in returned_tokenized_valid:
    #     print(sentences, end = ' \n\n')

            






    






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
import matplotlib.pyplot as plt
import seaborn as sns
import time

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



def _collect_layer_attn(model):
    """
    Returns a list of attention tensors (one per decoder layer) with shape (1, heads, seq, seq).
    Missing layers are skipped.
    """
    attn_list = []
    if hasattr(model, "layers"):
        for layer in model.layers:
            mha = getattr(layer, "self_attn", None)
            attn = getattr(mha, "last_attn", None)
            if attn is not None:
                attn_list.append(attn)  # (B, H, S, S)
    return attn_list

def _save_attention_heatmaps(attn_list, tokens, out_dir, tag="sample"):
    """
    Save a heatmap per layer and per head; also save per-layer averaged heads.
    attn_list: list of tensors with shape (1, heads, seq, seq)
    tokens: list of token strings of length seq
    """
    # print("WITHIN THE SAVE ATTENTION HEATMAPS FUNCTION...")
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, attn in enumerate(attn_list):
        print(f"Saving attention heatmaps for layer {layer_idx}...")
        # Ensure on CPU and numpy
        attn_np = attn[0].detach().cpu().numpy()  # (heads, seq, seq)
        num_heads, seq_len, _ = attn_np.shape
        labels = tokens[:seq_len]

        # Each head
        for head_idx in range(num_heads):
            print(f"        Saving attention heatmaps for head {head_idx}...")
            plt.figure(figsize=(min(20, max(6, seq_len/2)), min(20, max(6, seq_len/2))))
            sns.heatmap(attn_np[head_idx], xticklabels=labels, yticklabels=labels,
                        cmap="viridis", cbar=True, square=True, vmin=0.0, vmax=1.0)
            plt.title(f"{tag} - Layer {layer_idx} Head {head_idx}")
            plt.xlabel("Key (attended to)")
            plt.ylabel("Query (attending)")
            plt.tight_layout()
            print(f"        SAVING AT {os.path.join(out_dir, f'{tag}_layer{layer_idx}_head{head_idx}.png')}")
            plt.savefig(os.path.join(out_dir, f"{tag}_layer{layer_idx}_head{head_idx}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

        # Average across heads
        avg = attn_np.mean(axis=0)
        plt.figure(figsize=(min(20, max(6, seq_len/2)), min(20, max(6, seq_len/2))))
        sns.heatmap(avg, xticklabels=labels, yticklabels=labels,
                    cmap="viridis", cbar=True, square=True, vmin=0.0, vmax=1.0)
        plt.title(f"{tag} - Layer {layer_idx} Average over Heads")
        plt.xlabel("Key (attended to)")
        plt.ylabel("Query (attending)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_layer{layer_idx}_avg.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def _sample_next_token(logits, temperature=0.5, top_k=0, stochastic=True):
    """
    Takes logits (shape [1, vocab_size]) and returns a sampled token id
    and its log probability.
    """
    if temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        values, indices = torch.topk(logits, top_k, dim=-1)
        kth = values[:, -1].unsqueeze(-1)
        mask = logits < kth
        logits = logits.masked_fill(mask, -float('Inf'))

    # Use log_softmax for numerical stability in probs
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    if stochastic:
        next_id = int(torch.multinomial(probs, num_samples=1).item())
    else:
        next_id = int(torch.argmax(probs, dim=-1).item())
    
    token_log_prob = log_probs[0, next_id].item()
    
    return next_id, token_log_prob


def generate(input_prompt: str,
             device: str,
             model, vocab, max_length,
             max_output_length: int = 100,
             temperature = 0.5, top_k = 20, 
             stochastic = True,
             reference_text = None,
             visualize_attn: bool = True,
             attn_output_dir: str = '/home/anwesh/ELL8299 Project/attn_heatmaps/',
             sample_tag = 'sample',
             use_cache = True) -> dict:

    
    prompt_tensor = prepare_prompt(input_prompt, max_length, vocab)
    prompt_tensor = prompt_tensor.to(torch.device(device))

    model.to(device)
    model.eval()

    generated_tokens = []
    token_log_probs = []
    past_kv_cache = None

    with torch.no_grad():

        if use_cache:
            
            #process the prompt to initialize the cache
            logits, past_kv_cache = model(prompt_tensor, past_kv_cache=None)
            next_token_logits = logits[:, -1, :] 
            
            #sample first token
            next_id, log_prob = _sample_next_token(next_token_logits, temperature, top_k, stochastic)

            #new input is single new token
            input_tensor = torch.tensor([[next_id]], device=torch.device(device))  # (1, 1) shape

            generated_tokens.append(vocab.get_itos()[next_id])
            token_log_probs.append(log_prob)

            for _ in range(max_output_length - 1):
                current_seq_len = past_kv_cache[0][0].size(2)  # Get current sequence length from cache

                if current_seq_len + 1 >= model.positional_encoding.pe.size(1):
                    # if input length exceeds model's max positional encoding length, break
                    break   
            

                logits, past_kv_cache = model(input_tensor, mask = None, past_kv_cache=past_kv_cache)        # (1, seq_len, vocab_size)
                
                ###CHECK HERE
                next_token_logits = logits.squeeze(0)  # (vocab_size)

                next_id, log_prob = _sample_next_token(next_token_logits, temperature, top_k, stochastic)
                
                if next_id == vocab['<eos>']:
                    break

                generated_tokens.append(vocab.get_itos()[next_id])
                token_log_probs.append(log_prob)
                input_tensor = torch.tensor([[next_id]], device=torch.device(device))  # (1, 1) shape

            final_tensor = torch.cat([prompt_tensor] + [torch.tensor([[vocab[t] for t in generated_tokens if t in vocab]] ,device=device)], dim=1)

        else:    
            
            input_tensor = prompt_tensor  # (1, seq_len)
        
            for _ in range(max_output_length):
                if input_tensor.size(1) >= model.positional_encoding.pe.size(1):
                    break
            

                logits, _ = model(input_tensor, mask = None, past_kv_cache = None)        # (1, seq_len, vocab_size
                
                next_token_logits = logits[:, -1, :]  # (1, vocab_size
                
                next_id, log_prob = _sample_next_token(next_token_logits, temperature, top_k, stochastic)

                if next_id == vocab['<eos>']:
                    break

                generated_tokens.append(vocab.get_itos()[next_id])
                token_log_probs.append(log_prob)

                next_tensor = torch.tensor([[next_id]], device=torch.device(device)) #  -> (1, 1) shape
                input_tensor = torch.cat([input_tensor, next_tensor], dim=1)  #

            final_tensor = input_tensor


    mean_perplexity = None
    if len(token_log_probs) > 0:
        perplexity_per_token = [math.exp(-log_prob) for log_prob in token_log_probs]
        mean_perplexity = sum(perplexity_per_token)/len(perplexity_per_token)
    
    bleu = None
    if reference_text:
        
        reference_text = ' '.join(reference_text).lower()

        reference_tokens = [spacy_tokenize(reference_text.lower())]
        references = [reference_tokens]

        bleu = bleu_score([generated_tokens], references)

    if visualize_attn:
        with torch.no_grad():
            _ = model(final_tensor, mask=None, past_kv_cache=None) 
        
        itos = vocab.get_itos()
        token_ids = final_tensor[0].tolist()
        tokens = [itos[token_id] for token_id in token_ids]
        attn_list = _collect_layer_attn(model)

        if(len(attn_list) > 0 and (attn_output_dir is not None)):
            _save_attention_heatmaps(attn_list, tokens, attn_output_dir, tag=sample_tag)

    generated_text = ' '.join(generated_tokens)
    return generated_text, mean_perplexity, bleu


def generate_from_tinystories(args, model, params, num_samples = 5, 
                              tokenized_validation = tokenized_valid, 
                              beam_search: bool = False, beam_size : int = 10,
                              attn_output_dir: str = '/home/anwesh/ELL8299 Project/attn_heatmaps/',
                              kv_cache_use = True):
   
   ##Throw error if kv_cache_use and beam_search both true
   if kv_cache_use and beam_search:
       raise ValueError("KV Cache cannot be used with Beam Search. Please set kv_cache_use to False when using Beam Search.")

   print("KV Cache use is set to : ", kv_cache_use, '\n\n')
   
    
   #Sample a random examples from validation set
   trimmed_tokenized_valid = np.random.choice(tokenized_validation, num_samples, replace = False)

   ground_truth_completions = [example['tokens'][5:] for example in trimmed_tokenized_valid]
   tiny_stories_prompts = [example['tokens'][:5] for example in trimmed_tokenized_valid]
   perplexity_per_token_list = []
   bleu_list = []
   generated_completions = []

   avg_time_per_token = []

   for prompts, ground_truth in zip(tiny_stories_prompts, ground_truth_completions):
        prompt_text = ' '.join(prompts)

        if beam_search:
            
            start_time = time.time()

            generated_text, perplexity_per_token , bleu = beam_search_generate(input_prompt= prompt_text, model = model, device = args.device, 
                                                                vocab = tiny_stories_vocab, max_output_length = args.max_output_length, 
                                                                max_length = params['seq_len'], beam_size = beam_size,
                                                                length_penalty = 0.8, reference_text = ground_truth)
            end_time = time.time()
        


        else:

            start_time = time.time()

            generated_text, perplexity_per_token, bleu = generate(input_prompt = prompt_text, model = model, device = args.device, 
                                                              vocab = tiny_stories_vocab, max_output_length = args.max_output_length, 
                                                              max_length = params['seq_len'], temperature = args.temperature,
                                                              top_k = args.top_k, stochastic = args.stochastic,
                                                              reference_text = ground_truth, attn_output_dir=attn_output_dir,
                                                              use_cache=kv_cache_use)

        end_time = time.time()
        avg_time_per_token.append((end_time - start_time)/len(spacy_tokenize(generated_text)))
        perplexity_per_token_list.append(perplexity_per_token)
        bleu_list.append(bleu)
        generated_completions.append(spacy_tokenize(generated_text))
        

   mean_perplexity = sum(perplexity_per_token_list)/len(perplexity_per_token_list)
   mean_bleu = sum(bleu_list)/len(bleu_list)
   mean_time_per_token = sum(avg_time_per_token)/len(avg_time_per_token)

   print("Average Perplexity per token on samples from TinyStories validation set: ", mean_perplexity)
   print(f"Average BLEU score on {num_samples} samples from TinyStories validation set: ", mean_bleu)
   
   if beam_search:
        print(f"Average time taken per token with Beam Search (k = {beam_size}): ", mean_time_per_token)
   else:
        print("Average time taken per token with Sampling/Greedy Decoding: ", mean_time_per_token)


   return (tiny_stories_prompts, generated_completions), perplexity_per_token_list, bleu_list, avg_time_per_token
    
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


class BeamSearchHelper:
    """Helper class to store beam search hypotheses."""
    def __init__(self, tensor, score, cache):
        self.tensor = tensor  # full sequence of token IDs
        self.score = score    # cumulative log-prob
        self.cache = cache    # past_kv_cache for this beam


def beam_search_generate(input_prompt: str,
                         device: str,
                         model, vocab, max_length,
                         beam_size: int = 5,
                         max_output_length: int = 100,
                         length_penalty: float = 1.0, 
                         reference_text = None) -> dict:
    
    prompt_tensor = prepare_prompt(input_prompt, max_length, vocab)
    prompt_tensor = prompt_tensor.to(torch.device(device))
    prompt_length = prompt_tensor.size(1)

    model.to(device)
    model.eval()
    eos_id = vocab['<eos>']

    with torch.no_grad():
        #priming the cache with the prompt
        logits, past_kv_cache = model(prompt_tensor, past_kv_cache=None)
        next_token_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        
        top_log_probs, top_indices = torch.topk(next_token_log_probs, beam_size, dim=-1)

        active_beams = []
        completed_beams = []

        for i in range(beam_size):
            next_token_id = int(top_indices[0, i].item())
            log_prob = float(top_log_probs[0, i].item())
            
            # This tensor now stores the *full* sequence
            new_tensor = torch.cat([prompt_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            if next_token_id == eos_id:
                completed_beams.append(BeamSearchHelper(new_tensor, log_prob, past_kv_cache))
            else:
                active_beams.append(BeamSearchHelper(new_tensor, log_prob, past_kv_cache))


        # generation loop

        for _ in range(max_output_length - 1):
            if not active_beams:
                break
            
            all_candidates = []
            new_cache_map = {} # This will store the *new* caches

            for beam_idx, beam in enumerate(active_beams):
                
                # Check total length
                current_seq_len = beam.tensor.size(1)
                if current_seq_len >= model.positional_encoding.pe.size(1):
                    completed_beams.append(beam)
                    continue
                
                #last token to feed into the model
                input_tensor = beam.tensor[:, -1].unsqueeze(0) # Shape [1, 1]
                
                # ruun model with this beam and cache
                logits, new_cache = model(input_tensor, past_kv_cache=beam.cache)
                
                # Store the new cache
                new_cache_map[beam_idx] = new_cache

                next_token_log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
                top_log_probs, top_indices = torch.topk(next_token_log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    next_token_id = int(top_indices[0, i].item())
                    next_log_prob = float(top_log_probs[0, i].item())

                    new_score = beam.score + next_log_prob
                    new_tensor = torch.cat([beam.tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
                    
                    # Temporarily store the parent beam's index in the cache field
                    temp_beam = BeamSearchHelper(new_tensor, new_score, beam_idx)

                    if next_token_id == eos_id:
                        completed_beams.append(temp_beam)
                    else:
                        all_candidates.append(temp_beam)
            
            # --- 3. Prune and Re-order Caches ---
            if not all_candidates:
                active_beams = []
                continue

            all_candidates.sort(key=lambda x: x.score / (x.tensor.size(1) ** length_penalty), reverse=True)
            
            next_active_beams = []
            for candidate in all_candidates[:beam_size]:
                # Get the parent beam's index
                parent_idx = candidate.cache
                # Get the *new* cache generated by that parent
                actual_cache = new_cache_map[parent_idx]
                # Create the final beam object for the next loop
                candidate.cache = actual_cache
                next_active_beams.append(candidate)
            
            active_beams = next_active_beams
                
        completed_beams.extend(active_beams)

    # --- 4. Find Best Beam ---
    if not completed_beams:
        return "", float('inf'), 0.0
    else:
        completed_beams.sort(key=lambda x: x.score / (x.tensor.size(1) ** length_penalty), reverse=True)
        best_beam = completed_beams[0]
    
    all_token_ids = best_beam.tensor[0].tolist()
    generated_token_ids = all_token_ids[prompt_length:]
    
    # Filter out EOS tokens if they are in the middle
    final_tokens = []
    for token_id in generated_token_ids:
        if token_id == eos_id:
            break
        final_tokens.append(vocab.get_itos()[token_id])
    generated_text = ' '.join(final_tokens)

    # --- 5. Post-processing ---
    mean_perplexity = None
    num_generated = len(final_tokens)
    if num_generated > 0:
        # Use average log-prob for perplexity
        avg_log_prob = best_beam.score / num_generated
        mean_perplexity = math.exp(-avg_log_prob)

    bleu = None
    if reference_text:
        reference_text = ' '.join(reference_text).lower()
        reference_tokens = [spacy_tokenize(reference_text.lower())]
        references = [reference_tokens]
        bleu = bleu_score([final_tokens], references)

    # (Visualization logic removed for simplicity, as it requires another full pass)
    
    return generated_text, mean_perplexity, bleu



# def beam_search_generate(input_prompt: str,
#                          device: str,
#                          model, vocab, max_length,
#                          beam_size: int = 5,
#                          max_output_length: int = 100,
#                          length_penalty: float = 1.0, 
#                          reference_text = None) -> dict:
    

    
    
#     input_tensor = prepare_prompt(input_prompt, max_length, vocab)
#     input_tensor = input_tensor.to(torch.device(device))

#     model.to(device)
#     model.eval()

#     eos_id = vocab['<eos>']
#     init_beams = BeamSearchHelper(input_tensor, 0.0)  # Initialize beams
#     active_beams = [init_beams]
#     completed_beams = []

#     with torch.no_grad():

#         for _ in range(max_output_length):

#             if input_tensor.size(1) >= model.positional_encoding.pe.size(1):
#                 #if input length exceeds model's max positional encoding length, break
#                 break

#             if not active_beams:
#                 break  # All beams have completed
            
#             all_candidates = []

#             for beam in active_beams:
#                 if beam.tensor.size(1) >= model.positional_encoding.pe.size(1):
#                     completed_beams.append(beam)
#                     continue

                    
#                 logits = model(beam.tensor)        # (1, seq_len, vocab_size)
                
#                 next_token_log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab_size)

#                 top_log_probs, top_indices = torch.topk(next_token_log_probs, beam_size, dim=-1)

#                 for i in range(beam_size):
#                     next_token_id = int(top_indices[0, i].item())
#                     next_log_prob = float(top_log_probs[0, i].item())

#                     new_score = beam.score + next_log_prob
#                     new_tensor = torch.cat([beam.tensor, torch.tensor([[next_token_id]], device=torch.device(device))], dim=1)

#                     if next_token_id == eos_id:
#                         completed_beams.append(BeamSearchHelper(new_tensor, new_score))
#                     else:
#                         all_candidates.append(BeamSearchHelper(new_tensor, new_score))
            

#             if all_candidates:
#                 all_candidates.sort(key=lambda x: x.score / (x.tensor.size(1) ** length_penalty), reverse=True)
#                 active_beams = all_candidates[:beam_size]
#             else:
#                 active_beams = []
                
#         completed_beams.extend(active_beams)

#     if not completed_beams:
#         return "", float('inf'), 0.0  # No valid beams generated
#     else:
#         completed_beams.sort(key=lambda x: x.score / (x.tensor.size(1) ** length_penalty), reverse=True)
#         best_beam = completed_beams[0]
    

#     prompt_length = input_tensor.size(1)
#     all_token_ids = best_beam.tensor[0].tolist()
#     generated_token_ids = all_token_ids[prompt_length:]


#     generated_tokens = [vocab.get_itos()[token_id] for token_id in generated_token_ids]
#     generated_text = ' '.join(generated_tokens)

#     mean_perplexity = None
#     num_generated = len(generated_token_ids)
    
#     if num_generated > 0:
#         log_probs = [best_beam.score / num_generated] * num_generated
#         perplexity_per_token = [math.exp(-log_prob) for log_prob in log_probs]
#         mean_perplexity = sum(perplexity_per_token)/len(perplexity_per_token)

#     bleu = None
#     if reference_text:
        
#         reference_text = ' '.join(reference_text).lower()

#         reference_tokens = [spacy_tokenize(reference_text.lower())]
#         references = [reference_tokens]

#         bleu = bleu_score([generated_tokens], references)

#     if tokenized_valid:
#         _ = model(input_tensor)
#         itos = vocab.get_itos()
#         token_ids = best_beam.tensor[0].tolist()
#         tokens = [itos[token_id] for token_id in token_ids]

#     return generated_text, mean_perplexity, bleu


            

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str,
                        default = '/home/anwesh/scratch/ELL8299 Project/final_run_ckpts/seq64_layers3_heads8_lr0.0003_wd0.0/best_model.pt',
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
    parser.add_argument('--kv_cache', default = True, type = bool)

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

        (trimmed_prompts, generated_texts), perplexity_list, bleu_list, avg_time_per_token = generate_from_tinystories(args, model, params, 
                                                                                                                       tokenized_validation = tokenized_valid, 
                                                                                                                       beam_search = False, attn_output_dir = None)


        for items in zip(trimmed_prompts, generated_texts, perplexity_list, bleu_list, avg_time_per_token):
            prompt = ' '.join(items[0])
            generated = ' '.join(items[1])
            perplexity = items[2]
            bleu = items[3]
            avg_time_per_token = items[4]

            print(f"\nPrompt: {prompt}\nGenerated Text: {generated}\nPerplexity per token: {perplexity}\nBLEU Score: {bleu}\nAvg time per token: {avg_time_per_token}\n\n") 


    

    


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

            






    






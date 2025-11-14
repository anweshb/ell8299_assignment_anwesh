from train import *
from model import *
import torch
import wandb
import time
import json


def checkpointing_experiment():
    """
    Compare training with and without gradient checkpointing.
    Measures peak memory usage and epoch duration.
    """
    
    set_seed()
    
    # Configuration
    config = {
        'seq_len': 64,
        'num_layers': 6,  # Use more layers to see bigger memory difference
        'num_heads': 8,
        'lr': 3e-4,
        'weight_decay': 0,
        'num_epochs': 10,
        'batch_size': 128,
        'subset_size': 100_000,
        'device': 'cuda:1'
    }
    
    print("Checkpointing Experiment Configuration:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Create dataset subsets
    print("\nCreating dataset subsets...")
    train_subset = tokenized_train.select(range(min(config['subset_size'], len(tokenized_train))))
    valid_subset = tokenized_valid.select(range(min(10_000, len(tokenized_valid))))
    
    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(valid_subset)}")
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        train_subset, 
        valid_subset, 
        tiny_stories_vocab, 
        config['seq_len'], 
        batch_size=config['batch_size']
    )
    
    # Store results
    results = {}


    # ===== EXPERIMENT 2: WITH CHECKPOINTING =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Training WITH Gradient Checkpointing")
    print("=" * 60)
    
    # Initialize wandb for with checkpointing
    wandb.init(
        project="decoder-transformer-checkpointing",
        name=f"with_checkpoint_layers{config['num_layers']}_bs{config['batch_size']}",
        config={**config, 'use_checkpointing': True},
        reinit=True
    )
    
    # Create new model
    embedding_layer = create_embedding_layer()
    model_with_ckpt = create_decoder_transformer(
        seq_len=config['seq_len'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        device=config['device'],
        pretrained_embeddings=embedding_layer
    )
    
    # Train WITH checkpointing
    results['with_checkpointing'] = train_with_checkpointing(
        train_loader,
        valid_loader,
        model_with_ckpt,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        num_epochs=config['num_epochs']
    )
    
    wandb.finish()
    
    # Clean up
    del model_with_ckpt
    
    # ===== EXPERIMENT 2: WITHOUT CHECKPOINTING =====
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Training WITHOUT Gradient Checkpointing")
    print("=" * 60)
    
    # Initialize wandb for without checkpointing
    wandb.init(
        project="decoder-transformer-checkpointing",
        name=f"no_checkpoint_layers{config['num_layers']}_bs{config['batch_size']}",
        config={**config, 'use_checkpointing': False},
        reinit=True
    )
    
    # Create model
    embedding_layer = create_embedding_layer()
    model_no_ckpt = create_decoder_transformer(
        seq_len=config['seq_len'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        device=config['device'],
        pretrained_embeddings=embedding_layer
    )
    
    # Train WITHOUT checkpointing (using standard training)
    results['without_checkpointing'] = train_standard(
        train_loader,
        valid_loader,
        model_no_ckpt,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        num_epochs=config['num_epochs']
    )
    
    wandb.finish()
    
    # Clean up
    del model_no_ckpt
    torch.cuda.empty_cache()
    time.sleep(5)  # Give GPU time to clear
    
    
    torch.cuda.empty_cache()
    
    # ===== PRINT COMPARISON =====
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print("\nWithout Checkpointing:")
    print(f"  Peak Memory: {results['without_checkpointing']['peak_memory_mb']:.2f} MB")
    print(f"  Avg Epoch Time: {results['without_checkpointing']['avg_epoch_time']:.2f} s")
    
    print("\nWith Checkpointing:")
    print(f"  Peak Memory: {results['with_checkpointing']['peak_memory_mb']:.2f} MB")
    print(f"  Avg Epoch Time: {results['with_checkpointing']['avg_epoch_time']:.2f} s")
    
    memory_saved_mb = results['without_checkpointing']['peak_memory_mb'] - results['with_checkpointing']['peak_memory_mb']
    memory_saved_pct = (memory_saved_mb / results['without_checkpointing']['peak_memory_mb']) * 100
    
    time_overhead = results['with_checkpointing']['avg_epoch_time'] - results['without_checkpointing']['avg_epoch_time']
    time_overhead_pct = (time_overhead / results['without_checkpointing']['avg_epoch_time']) * 100
    
    print("\nSavings:")
    print(f"  Memory Saved: {memory_saved_mb:.2f} MB ({memory_saved_pct:.1f}%)")
    print(f"  Time Overhead: +{time_overhead:.2f} s ({time_overhead_pct:.1f}%)")
    
    # Save results to JSON
    results_file = "/home/anwesh/scratch/ELL8299 Project/checkpointing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def train_standard(train_loader, validation_loader, decoder_transformer,
                   lr=3e-4, weight_decay=0, num_epochs=10):
    """
    Train WITHOUT checkpointing (standard forward/backward).
    ##We write this function again avoid wandb and ckpt clashes fromt train.py train_decoder_transformer.py
    Measures peak memory usage and epoch duration.
    Returns a dict with 'peak_memory_mb', 'avg_epoch_time', and 'epoch_times
    """
    
    optimizer = optim.Adam(decoder_transformer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tiny_stories_vocab['<pad>'])
    device = next(decoder_transformer.parameters()).device
    
    print(f"Starting standard training (no checkpointing).")
    
    epoch_times = []
    peak_memory_mb = 0
    
    for epoch in range(num_epochs):
        decoder_transformer.train()
        total_train_loss = 0
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        epoch_start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_seq = batch['target_ids'].to(device)
            
            # Standard forward pass (no checkpointing)
            output, _ = decoder_transformer(input_ids)
            
            output_flat = output.view(-1, output.size(-1))
            target_flat = target_seq.view(-1)
            loss = criterion(output_flat, target_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_transformer.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Track peak memory
        current_peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_memory_mb = max(peak_memory_mb, current_peak)
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_perplexity = float(np.exp(avg_train_loss))
        
        # Validation
        decoder_transformer.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                target_seq = batch['target_ids'].to(device)
                
                output, _ = decoder_transformer(input_ids)
                output_flat = output.view(-1, output.size(-1))
                target_flat = target_seq.view(-1)
                loss = criterion(output_flat, target_flat)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(validation_loader)
        val_perplexity = float(np.exp(avg_val_loss))
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Duration: {epoch_duration:.2f}s | Peak GPU Memory: {current_peak:.2f} MB")
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'epoch_duration_sec': epoch_duration,
            'peak_gpu_memory_mb': current_peak
        })
    
    avg_epoch_time = np.mean(epoch_times)
    
    return {
        'peak_memory_mb': peak_memory_mb,
        'avg_epoch_time': avg_epoch_time,
        'epoch_times': epoch_times
    }


if __name__ == "__main__":
    results = checkpointing_experiment()
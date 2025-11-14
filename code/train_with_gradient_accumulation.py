from train import *
from model import *
import wandb


def grad_accm_experiments():

    set_seed()

    config={
        'seq_len': 64,
        'num_layers': 3,
        'num_heads': 8,
        'lr': 3e-4,
        'weight_decay':0,
        'num_epochs': 5,
        'wandb_project': 'decoder-transformer-grad-accm',
        'accumulation_steps':8,
        'mini_batch_size': 16
    }
    
    print("Current configfuration:")

    for key, value in config.items():
        print(f"{key}: {value}")
    

    run_name = f"GRADACCM_{config['accumulation_steps']}_seq{config['seq_len']}_layers{config['num_layers']}_heads{config['num_heads']}_lr{config['lr']}_wd{config['weight_decay']}"
    wandb_dir = "/home/anwesh/scratch/ELL8299 Project/wandb"
    wandb_run_id = None

    tokenized_train_subset = tokenized_train.select(range(100_000))  # Using a small subset to make comparisons
    tokenized_valid_subset = tokenized_valid.select(range(10_000))  # For testing

    if wandb_run_id:
        wandb.init(project=config["wandb_project"],
                   id = wandb_run_id,
                   resume="allow",
                   name = run_name,
                   config=config,
                   dir = wandb_dir)
        
    else:
        wandb.init(project=config["wandb_project"],
                   name = run_name,
                   config=config,
                   dir = wandb_dir)


    print("Starting training with the following configuration:")
    
    for key, value in wandb.config.items():
        print(f"{key}: {value}")


    embedding_layer = create_embedding_layer()

    decoder_model = create_decoder_transformer(
        seq_len=config['seq_len'], 
        num_heads = config['num_heads'],
        num_layers = config['num_layers'],
        device='cuda:0',
        pretrained_embeddings=embedding_layer)

    train_loader, valid_loader = create_dataloaders(tokenized_train_subset, tokenized_valid_subset, 
                                                    tiny_stories_vocab, config['seq_len'], batch_size=config['mini_batch_size'])
    

    train_with_grad_accm(
        train_loader, 
        valid_loader, 
        decoder_transformer=decoder_model, 
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        num_epochs=config['num_epochs'], 
        accumulation_steps=config['accumulation_steps'])

    eval_decoder_transformer(
        decoder_transformer=decoder_model, 
        test_loader=valid_loader)

    wandb.finish()

if __name__ == "__main__":
    grad_accm_experiments()

from train import train_with_checkpointing
from train import *
from model import *



def checkpointing_experiment():

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
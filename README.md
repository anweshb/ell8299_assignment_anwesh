# 2025SIY257578 Individual Assignment

# Step 0:
### conda env create -f environment.yml


## Step 1:

### Go to tokenization.py, change paths for cache_location, and the download/save locations for embedding as well as vocab

## Step 2:

### Go to train.py, again change paths for loading, wandb etc based on your preference
### Make changes to training settings if from default args or pass args by running python train.py --desired_arg arg_value

## Step 3:

### Go to generate.py, specify saved model location, in the arg parser change temprature or specify custom prompt as per preference.
### By default TinyStories will be used. Set kv caching or beam search decoding as per preference from argparser

## Step 4:

### run train_with_gradient_accumulation.py

## Step 5:

### run train_with_checkpoint.py



# llmlens.c
This is a fork of Karpathy's llm.c in the spirit of Neel Nanda's TransformerLens. 
The current goal is to enable efficient, easy Sparse Autoencoder (SAE) training and feature in easy to read pure C/CUDA. Eventually, a webserver where a python notebook can make queries for inputs to visualize as features. I was motivated to make this because a lot of the current SAE training code is not very good and I think there may be a considerable amount of overhead in the way that current tools access the activations for training SAEs.

## quick start (GPU)

The "I don't care about anything I just want to train and I have a GPU" section. Run:

```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2fp32cu
./train_gpt2fp32cu
```

The above lines (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, (2) download and save the GPT-2 (124M) weights, (3) init from them in C/CUDA and train for one epoch on tineshakespeare with AdamW (using batch size 4, context length 1024, total of 74 steps), evaluate validation loss, and sample some text. Note that in this quickstart we are using the fp32 version [train_gpt2_fp32.cu](train_gpt2_fp32.cu) of the CUDA code. Below in the CUDA section we document the current "mainline" [train_gpt2.cu](train_gpt2.cu), which is still being very actively developed, uses mixed precision, and runs ~2X faster.

## quick start (multiple GPUs)

You'll be using the (more bleeding edge) mixed precision version of the code:

```
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2cu
mpirun -np <number of GPUs on your machine> ./train_gpt2cu
```

## license

MIT

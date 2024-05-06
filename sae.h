#pragma once

#include <cuda_runtime.h>

typedef struct {
    float *input_bias;
    float *encoder_weights;
    float *encoder_bias;
    float *decoder_weights;
    float *decoder_bias;
    int input_size;
    int hidden_size;
} SAE;

void initSAE(SAE *sae, int input_size, int hidden_size);
void freeSparseAE(SAE *sae);
void trainSparseAE(SAE *sae, float *data, int num_samples, int epochs, float learning_rate, float sparsity_target, float l1_penalty);
void forwardPass(SAE *sae, float *input, float *encoded, float *decoded);
void backwardPass(SAE *sae, float *input, float *encoded, float *decoded, float *gradInput, float *gradEncoded, float *gradDecoded, float sparsity_target, float l1_penalty, float learning_rate);

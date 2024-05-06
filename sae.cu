#include "sae.h"
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>

__global__ void applyReLU(float *input, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        input[id] = fmaxf(0.0, input[id]); // ReLU activation
    }
}

__global__ void addBiasAndActivate(float *data, float *bias, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        data[id] += bias[id % size];
        data[id] = fmaxf(0.0, data[id]); // ReLU activation
    }
}

__global__ void addBias(float *data, float *bias, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        data[id] += bias[id % size];
    }
}

__global__ void applyReluDerivative(float *grad, const float *orig, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        grad[id] *= orig[id] > 0 ? 1.0 : 0.0;
    }
}

void initSAE(SAE *sae, int input_size, int hidden_size) {
    cudaError_t cudaStatus;
    curandGenerator_t gen;

    // Create the generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1337); // Use a fixed seed for reproducibility

    // Allocate memory for weights and biases
    cudaMalloc(&sae->input_bias, input_size * sizeof(float));
    cudaMalloc(&sae->encoder_weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&sae->encoder_bias, hidden_size * sizeof(float));
    cudaMalloc(&sae->decoder_weights, hidden_size * input_size * sizeof(float));
    cudaMalloc(&sae->decoder_bias, input_size * sizeof(float));

    float mean = 0.0f;
    float stddev = 0.1f;
    curandGenerateNormal(gen,sae->input_bias, input_size, mean, stddev);
    curandGenerateNormal(gen, sae->encoder_weights, input_size * hidden_size, mean, stddev);
    curandGenerateNormal(gen, sae->encoder_bias, input_size, mean, stddev);
    curandGenerateNormal(gen, sae->decoder_weights, input_size * hidden_size, mean, stddev);
    curandGenerateNormal(gen, sae->decoder_bias, input_size, mean, stddev);

    // Clean up
    curandDestroyGenerator(gen);

    // Check for any errors during CUDA operations
    if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    }
}
// nvcc -o sae.exe sae.cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" -lcublas -lcurand
void freeSAE(SAE *sae) {
    cudaFree(&sae->input_bias);
    cudaFree(&sae->encoder_weights);
    cudaFree(&sae->encoder_bias);
    cudaFree(&sae->decoder_weights);
    cudaFree(&sae->decoder_bias);
}

void forwardPass(cublasHandle_t handle, SAE *sae, float *input, float *encoded, float *decoded) {
    int input_size = sae->input_size;
    int hidden_size = sae->hidden_size;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Input to Hidden (Encoder)
    // Note: cuBLAS assumes column-major storage, you might need to adjust if your data is row-major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, 1, input_size, &alpha, sae->encoder_weights, hidden_size, input, input_size, &beta, encoded, hidden_size);
    addBiasAndActivate<<<(hidden_size + 255) / 256, 256>>>(encoded, sae->encoder_bias, hidden_size);

    // Hidden to Output (Decoder)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_size, 1, hidden_size, &alpha, sae->decoder_weights, input_size, encoded, hidden_size, &beta, decoded, input_size);
    addBias<<<(input_size + 255) / 256, 256>>>(decoded, sae->decoder_bias, input_size);
}

void backwardPass(cublasHandle_t handle, SAE *sae, float *input, float *encoded, float *decoded, float *gradInput, float *gradEncoded, float *gradDecoded, float sparsity_target, float l1_penalty, float learning_rate) {
    int input_size = sae->input_size;
    int hidden_size = sae->hidden_size;
    float alpha = -learning_rate;  // Used for weight updates
    float beta = 1.0f;             // Used to accumulate gradients in existing weight matrices
    float neg_one = -1.0f;         // Used for calculating gradients
    float zero = 0.0f;
    // Gradient of the loss with respect to decoded output (dL/dy)
    cublasScopy(handle, input_size, input, 1.0f, gradDecoded, 1.0f);
    cublasSaxpy(handle, input_size, &neg_one, decoded, 1, gradDecoded, 1); // gradDecoded = input - decoded

    // Gradient of the loss with respect to encoder weights and biases (backpropagation)
    // Compute gradient at hidden layer (dL/dh)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_size, 1, input_size, &neg_one, sae->decoder_weights, input_size, gradDecoded, input_size, &zero, gradEncoded, hidden_size);
    applyReluDerivative<<<(hidden_size + 255) / 256, 256>>>(gradEncoded, encoded, hidden_size);

    // Update decoder weights (W2)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, input_size, hidden_size, 1, &alpha, gradDecoded, input_size, encoded, hidden_size, &beta, sae->decoder_weights, input_size);

    // Update encoder weights (W1)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_size, input_size, 1, &alpha, gradEncoded, hidden_size, input, input_size, &beta, sae->encoder_weights, hidden_size);

    // Update encoder bias (b1)
    cublasSaxpy(handle, hidden_size, &alpha, gradEncoded, 1, sae->encoder_bias, 1);

    // Update decoder bias (b2) - Sum the gradients for decoded output
    cublasSaxpy(handle, input_size, &alpha, gradDecoded, 1, sae->decoder_bias, 1);
}

void trainSAE(SAE *sae, float *data, int num_samples, int epochs, float learning_rate, float sparsity_target, float l1_penalty) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    //training loop

    cublasDestroy(handle);
}

int main() {
    printf("heyo");
    SAE sae;
    initSAE(&sae, 736, 736 * 8); //allocate memory for SAE
    int num_samples = 1000000;
    float* data = new float[num_samples](); //zero initialized dummy vector for now
    int epochs = 1000;
    float learning_rate = 0.01f;
    float sparsity_target = 0.05f; //voodoo idk why we choose these
    float l1_penalty = 0.0001f;
    trainSAE(&sae, data, num_samples, epochs, learning_rate, sparsity_target, l1_penalty);
    freeSAE(&sae); //destroy SAE
    return 0;
}
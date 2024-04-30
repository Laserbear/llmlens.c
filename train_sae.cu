#define TESTING
#include "train_gpt2.cu"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label, float threshold=1e-0) {
    // a is the calculated tensor, b is the reference tensor
    int print_upto = 10;
    int ok = 1;
    float max_diff = 0.0f;
    float max_rel_error = 0.0f;
    float max_a = 0.0f;
    float max_b = 0.0f;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            float denom = fabsf(b[i]);
            max_rel_error = (denom == 0.0f) ? 0.0f : diff / denom;
            max_a = a[i];
            max_b = b[i];
        }
        if (diff <= threshold) {
            if (i < print_upto) { printf("OK "); }
        } else {
            if (i < print_upto) { printf("NOT OK "); }
            ok = 0;
        }
        if (i < print_upto) { printf("%f %f\n", a[i], b[i]); }
    }
    // print the final result
    if (ok) {
        printf("TENSOR OK, max diff: %e, with rel error: %e (calculated=%f, ref=%f)\n",
                max_diff, max_rel_error, max_a, max_b);
    } else {
        printf("TENSOR NOT OK, max diff: %e, with rel error: %e (calculated=%f, ref=%f)\n",
                max_diff, max_rel_error, max_a, max_b);
    }
    return ok;
}

// the same tensors as in the train file, but in float, which are used as reference
typedef struct {
    float*  wte; // (Vp, C)
    float*  wpe; // (maxT, C)
    float*  ln1w; // (L, C)
    float*  ln1b; // (L, C)
    float*  qkvw; // (L, 3*C, C)
    float*  qkvb; // (L, 3*C)
    float*  attprojw; // (L, C, C)
    float*  attprojb; // (L, C)
    float*  ln2w; // (L, C)
    float*  ln2b; // (L, C)
    float*  fcw; // (L, 4*C, C)
    float*  fcb; // (L, 4*C)
    float*  fcprojw; // (L, C, 4*C)
    float*  fcprojb; // (L, C)
    float*  lnfw; // (C)
    float*  lnfb; // (C)
} FloatParameterTensors;
static_assert(sizeof(FloatParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

// malloc_and_point, but in float and on CPU, because we use this data to check correctness on CPU
float* float_cpu_malloc_and_point_parameters(FloatParameterTensors* params, size_t* param_sizes) {
    // calculate the total number of parameters
    size_t num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // everything is float so number of bytes to allocate is a simple multiplication
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

struct {
size_t width = 512;
size_t expansion_factor = 8; 
} SAEConfig

struct {
  SAEConfig config;

  ParameterTensors params;
  size_t param_elements[2];
  size_t param_sizeof[2];
  void* params_memory;
  size_t num_parameters;
  size_t num_parameters_bytes;
  //buffers for the AdamW optimizer
  float* m_memory;
  float* v_memory;

  //do i need to store inputs?
  float mean_loss;
  float accumulated_mean_loss; //support multi-gpu
  floatX* cpu_losses;
  unsigned long long rng_state;
} SAE

//TODO: custom SAE kernel fusing relu(A * B) * C

__global__ void backwardMultBias(const float *mat, const float *vec, const float *d_out, float *d_mat, float *d_vec, float *d_bias, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // Since bias affects the output directly and independently per row:
        atomicAdd(&d_bias[row], d_out[row]);

        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            float d_output = d_out[row];
            // Propagate d_out to weights: Gradient w.r.t. the weight is input scaled by d_out
            atomicAdd(&d_mat[idx], d_output * vec[col]);
            // Propagate d_out to input features: Gradient w.r.t. the input is weights scaled by d_out
            atomicAdd(&d_vec[col], d_output * mat[idx]);
        }
    }
}

__global__ void backwardBiasMultRelu(const float *mat, const float *vec, const float *bias, const float *d_out, float *d_mat, float *d_vec, float *d_bias, const float *out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float d_input = 0.0;
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            float grad_output = (out[row] > 0) ? d_out[row] : 0.0f; // ReLU derivative
            atomicAdd(&d_mat[idx], grad_output * vec[col]);
            atomicAdd(&d_vec[col], grad_output * mat[idx]);
            atomicAdd(&d_bias[col], grad_output);
        }
    }
}


__global__ void biasMultRelu(const float *mat, const float *vec, const float *bias, float *out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * (vec[col] + bias[col]); //test if repeating addition is faster than blocking on addition first
        }
        out[row] = max(0.0f, sum);  // Apply ReLU activation
    }
}

__global__ void multBias(const floatX *mat, const floatX *vec, const floatX *bias, float *out, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += mat[row * cols + col] * vec[col];
        }
        out[row] = sum + bias[row];  // Add bias right after multiplication
    }
}

void forward(floatX *input, floatX *bias_1, floatX *bias_2, floatX *W_1, floatX *W_2, int W_1_m, int W_1_n, int W_2_m, int W_2_n) {
    int threads = 256;
    int blocksW1m = (W_1_m + threads - 1) / threads; // should be 2
    int blocksW2m = (W_2_m + threads - 1) / threads;

    // Step 1: Apply bias_1, multiply by W_1 and apply ReLU
    biasMultRelu<<<blocksW1m, threads>>>(W_1, input, bias_1, scratch, W_1_m, W_1_n);

    // Step 2: Multiply scratch by W_2 and add bias_2
    multBias<<<blocksW2m, threads>>>(W_2, scratch, bias_2, output, W_2_m, W_1_n);
}

void sae_forward(SAE *sae, floatX* activations, size_t num_activations, size_t batch_size, size_t seq)length) {
  //ensure model was initialized properly
  if (sae->params_memory == NULL) {
    printf("Error: Autoencoder was not initialized properly. \n");
    exit(EXIT_FAILURE);
  }
  size_t expansion_factor = model->config.expansion_factor;
  size_t num_neurons = model->config.num_neurons;
  
  //encoder: bias -> linear -> relu 
  //decoder: linear -> bias
  

  cudaCheck(cudaMalloc((void**)inputs, num_activations * batch_size * seq_length * sizeof(floatX)));
  cudaCheck(cudaMalloc(void**)cpu_losses, batch_size * seq_length * sizeof(floatX));

  //TODO: alloc mem
  //run kernel
  //profit?

}

void sae_backward() {
   //todo
}


int main(int argc, char *argv[]) {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("[System]\n");
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    enable_tf32 = 0; // NOTE: disable TF32 for testing!!!
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");
    size_t V = model.config.vocab_size;
    size_t Vp = model.config.padded_vocab_size;
    size_t maxT = model.config.max_seq_len;
    size_t L = model.config.num_layers;
    size_t C = model.config.channels;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { fprintf(stderr, "Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    assert(0 <= T && T <= maxT);
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    // TODO: train SAE!


    // free everything
    free(x);
    free(y);
    free(logits_cpu_raw);
    free(logits_cpu);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    free(grads_memory_cpu);
    free(grads_memory_cpu_float);
    gpt2_free(&model);
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));

    return 0;
}

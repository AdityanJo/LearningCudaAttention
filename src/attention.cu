
/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>
#include <tuple>

void readFile(float *arr, FILE *handle, const char *weight_name,
              char *filePath) {
  handle = fopen(filePath, "rb");

  if (handle == NULL) {
    printf("[E] Unable to locate %s file, ensure specified path is correct\n",
           weight_name);
    exit(-1);
  }

  // From
  // https://stackoverflow.com/questions/1422817/how-to-read-a-float-from-binary-file-in-c
  fseek(handle, 0, SEEK_END);
  long size = ftell(handle);
  fseek(handle, 0, SEEK_SET);

  printf("[I] [Weight %s] [Attempting load size: %ld]\n", weight_name, size);

  if (fread(arr, sizeof(float), size, handle) != size) {
    printf("[I] Loaded %s : [%ld] \n", weight_name, size);
    // printf("%f\n",arr[0]);
    fclose(handle);
  } else {
    printf("[E] Unable to load %s successfully [Attempted load size: %ld]\n",
           weight_name, size);
  }
}
std::tuple<float *, float *, float *, float *, float *, float *, float *>
allocateAndLoadHostMemory(char *inputsFile, char *queryFile, char *keyFile,
                          char *valueFile, char *outWtFile, char *outBiasFile,
                          char *outputFile, int dim, int heads, int dim_head,
                          int seq_len) {
  float *hInputs;
  float *hQuery;
  float *hKey;
  float *hValue;
  float *hOutputWt;
  float *hOutputBias;
  float *hOutputs;

  hInputs = (float *)malloc(sizeof(float) * seq_len * dim);
  hQuery = (float *)malloc(sizeof(float) * heads * dim_head * dim);
  hKey = (float *)malloc(sizeof(float) * heads * dim_head * dim);
  hValue = (float *)malloc(sizeof(float) * heads * dim_head * dim);
  hOutputWt = (float *)malloc(sizeof(float) * seq_len * dim);
  hOutputBias = (float *)malloc(sizeof(float) * seq_len);
  hOutputs = (float *)malloc(sizeof(float) * seq_len * dim);

  FILE *fp;

  readFile(hInputs, fp, "inputs", inputsFile);
  readFile(hQuery, fp, "query", queryFile);
  readFile(hKey, fp, "key", keyFile);
  readFile(hValue, fp, "value", valueFile);
  readFile(hOutputWt, fp, "out_wt", outWtFile);
  readFile(hOutputBias, fp, "out_bias", outBiasFile);
  readFile(hOutputs, fp, "outputs", outputFile);

  return {hInputs, hQuery, hKey, hValue, hOutputWt, hOutputBias, hOutputs};
}

void deallocateHostMemory(float *hInputs, float *hQuery, float *hKey,
                          float *hValue, float *hOutputWt, float *hOutputBias,
                          float *hOutputs) {
  free(hInputs);
  free(hQuery);
  free(hKey);
  free(hValue);
  free(hOutputWt);
  free(hOutputBias);
  free(hOutputs);
}

std::tuple<float *, float *, float *, float *, float *, float *, float *>
allocateAndCopyDeviceMemory(float *hInputs, float *hQuery, float *hKey,
                            float *hValue, float *hOutputWt, float *hOutputBias,
                            float *hOutputs, int dim, int heads, int dim_head,
                            int seq_len) {
  float *dInputs, *dQuery, *dKey, *dValue, *dOutputWt, *dOutputBias, *dOutputs;
  cudaMalloc((void **)&dInputs, sizeof(float) * seq_len * dim);
  cudaMalloc((void **)&dQuery, sizeof(float) * heads * dim_head * dim);
  cudaMalloc((void **)&dKey, sizeof(float) * heads * dim_head * dim);
  cudaMalloc((void **)&dValue, sizeof(float) * heads * dim_head * dim);
  cudaMalloc((void **)&dOutputWt, sizeof(float) * seq_len * dim);
  cudaMalloc((void **)&dOutputBias, sizeof(float) * seq_len);
  cudaMalloc((void **)&dOutputs, sizeof(float) * seq_len * dim);

  printf("[I] Allocated device memory\n");

  cudaMemcpy(dInputs, hInputs, sizeof(float) * seq_len * dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dQuery, hQuery, sizeof(float) * heads * dim_head * dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dKey, hKey, sizeof(float) * heads * dim_head * dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dValue, hValue, sizeof(float) * heads * dim_head * dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dOutputWt, hOutputWt, sizeof(float) * seq_len * dim,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dOutputBias, hOutputBias, sizeof(float) * seq_len,
             cudaMemcpyHostToDevice);
  // cudaMemcpy(dOutputs, hOutputs, sizeof(float)*seq_len*dim,
  // cudaMemcpyHostToDevice);
  printf("[I] Copied host to device memory\n");

  return {dInputs, dQuery, dKey, dValue, dOutputWt, dOutputBias, dOutputs};
}

void deallocateDeviceMemory(float *dInputs, float *dQuery, float *dKey,
                            float *dValue, float *dOutputWt, float *dOutputBias,
                            float *dOutputs) {
  cudaFree(dInputs);
  cudaFree(dQuery);
  cudaFree(dKey);
  cudaFree(dValue);
  cudaFree(dOutputWt);
  cudaFree(dOutputBias);
  cudaFree(dOutputs);
}

__global__ void print_tensor(int n, float *x, bool rev = false) {
  if (rev) {
    for (int i = n; i > n - 10; i--) {
      printf("%f, ", x[i]);
    }

  } else {
    for (int i = 0; i < n; i++) {
      printf("%f, ", x[i]);
    }
  }
  printf("\n");
}

void forward_fc(cublasHandle_t handle, float *inputs, float *kernel,
                float *bias, float *outputs, int m, int n, int k,
                char *layer_name, const float alpha, const float beta,
                cublasOperation_t transa, cublasOperation_t transb) {
  cublasStatus_t status = cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                      kernel, k, inputs, k, &beta, outputs, m);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[E] Error in FC(%s) : %s\n", layer_name,
           cublasGetStatusString(status));
    return;
  }
}

__device__ void convertRowMajorToColumnMajor(float *in, float *out, int row,
                                             int col) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < row * col) {
    int orig_row = i % row;
    int orig_col = int(i / row);
    out[orig_col * row + orig_row] = in[i];
  }
}

__global__ void qk_matmul(float *A, float *B, float *C, int m, int n, int k,
                          float alpha) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // A - m x k
  // B - k x n
  // C - m x n
  if (i < m && j < n) {
    float tmp = 0.0f;
    for (int l = 0; l < k; l++) {
      tmp += A[i * k + l] * B[j * n + l];
      // printf("%f x %f \n", A[i * k + l], B[j * n + l]);
    }
    C[i * n + j] = tmp * alpha;
  }
}

__global__ void attnval_matmul(float *A, float *B, float *C, int m, int n,
                               int k, float alpha) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // A - m x k
  // B - k x n
  // C - m x n
  if (i < m && j < n) {
    float tmp = 0.0f;
    for (int l = 0; l < k; l++) {
      tmp += A[i * k + l] * B[l * n + j];
      // printf("[I] %f x %f \n", A[i * k + l], B[l * n + j]);
    }
    C[i * n + j] = tmp;
  }
}

__global__ void qkv_matmul(float *A, float *B, float *C, int m, int n, int k,
                           float alpha) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // A - m x k
  // B - k x n
  // C - m x n
  if (i < m && j < n) {
    float tmp = 0.0f;
    // A i l x A l j
    for (int l = 0; l < k; l++) {
      tmp += A[i * k + l] * B[j * k + l];
      // printf("%f x %f \n", A[i * k + l], B[j * n + l]);
    }
    C[i * n + j] = tmp * alpha;
  }
}

__global__ void out_proj(float *A, float *B, float *C, float *D, int m, int n,
                         int k, float alpha) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // A - m x k
  // B - k x n
  // C - m x n
  if (i < m && j < n) {
    float tmp = 0.0f;
    // A i l x A l j
    for (int l = 0; l < k; l++) {
      tmp += A[i * k + l] * B[j * k + l];
      // printf("%f x %f | [%d x %d x %d x %d x %d x %d]\n", A[i * k + l], B[j *
      // k + l], i, j, l, m, n, k);
    }
    D[i * n + j] = tmp + C[j];
  }
}

void forward(float *dInputs, float *dQuery, float *dKey, float *dValue,
             float *dOutputWt, float *dOutputBias, float *dOutputs, int dim,
             int heads, int dim_head, int seq_len) {
  cudaDeviceSynchronize();
  int inner_dim = dim_head * heads;
  float scale = pow(dim_head, -0.5);

  const float alpha = 1.0f;
  const float alphaScale = scale;
  const float beta = 0.0f;

  float *dQ, *dK, *dV, *dDots, *dDotsCol, *dAttn, *dAttnOut;

  cublasHandle_t cublasHandle;
  cudnnHandle_t cudnnHandle;
  cublasCreate(&cublasHandle);
  cudnnCreate(&cudnnHandle);

  cudaMalloc((void **)&dQ, sizeof(float) * seq_len * inner_dim);
  cudaMalloc((void **)&dK, sizeof(float) * seq_len * inner_dim);
  cudaMalloc((void **)&dV, sizeof(float) * seq_len * inner_dim);
  cudaMalloc((void **)&dDots, sizeof(float) * seq_len * dim);
  cudaMalloc((void **)&dDotsCol, sizeof(float) * seq_len * dim);
  cudaMalloc((void **)&dAttn, sizeof(float) * seq_len * dim);
  cudaMalloc((void **)&dAttnOut, sizeof(float) * seq_len * inner_dim);
  cudaError_t err = cudaGetLastError();

  //  printf("%s\n", cudaGetErrorString(err));
  // op(A) -  m x k
  // op(B) - k x n
  // C - m x n
  // m, n, k =  seq_len, inner_dim, dim
  // lda, ldb, ldc = dim, dim, inner_dim
  // dInputs = m 1024 x k 1024
  // dQuery = k 1024 x n 512

  int blockSz = 16;
  dim3 gridDim((seq_len + blockSz - 1) / blockSz,
               (dim + blockSz - 1) / blockSz);
  dim3 blockDim(blockSz, blockSz);
  qkv_matmul<<<gridDim, blockDim>>>(dInputs, dQuery, dQ, dim, inner_dim,
                                    seq_len, alpha);
  qkv_matmul<<<gridDim, blockDim>>>(dInputs, dKey, dK, dim, inner_dim, seq_len,
                                    alpha);
  qkv_matmul<<<gridDim, blockDim>>>(dInputs, dValue, dV, dim, inner_dim,
                                    seq_len, alpha);

  //  // dQ - 1 x seq_len x inner_dim
  //  // dK - 1 x seq_len x inner_dim
  // convertRowMajorToColumnMajor(d, float *out, int row, int col)

  // cublasStatus_t status = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
  // dim, seq_len, inner_dim, &alphaScale, dQ, inner_dim, dK, inner_dim, &beta,
  // dDots, dim);

  qkv_matmul<<<gridDim, blockDim>>>(dQ, dK, dDots, dim, seq_len, inner_dim,
                                    alphaScale);

  cudnnTensorDescriptor_t inputDesc, outputDesc;
  cudnnCreateTensorDescriptor(&inputDesc);
  cudnnCreateTensorDescriptor(&outputDesc);

  cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             seq_len, dim, 1, 1);
  cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             seq_len, dim, 1, 1);

  cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, inputDesc, dDots,
                      &beta, outputDesc, dAttn);

  dim3 gridDim2((seq_len + blockSz - 1) / blockSz,
                (dim + blockSz - 1) / blockSz);
  attnval_matmul<<<gridDim2, blockDim>>>(dAttn, dV, dAttnOut, dim, inner_dim,
                                         seq_len, alpha);

  out_proj<<<gridDim, blockDim>>>(dAttnOut, dOutputWt, dOutputBias, dOutputs,
                                  dim, seq_len, inner_dim, alpha);
  cudaDeviceSynchronize();

  cublasDestroy(cublasHandle);
  cudnnDestroy(cudnnHandle);

  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);

  cudaDeviceSynchronize();
  cudaFree(dDotsCol);
  cudaFree(dDots);
  cudaFree(dAttn);
  cudaFree(dAttnOut);
  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
}

std::tuple<float> compareOutputs(float *hOutputs, float *dOutputs, int N) {
  float mse = 0.0f;
  float *hRefOutputs = (float *)malloc(sizeof(float) * N);
  cudaMemcpy(hRefOutputs, dOutputs, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    mse += pow(hRefOutputs[i] - hOutputs[i], 2);
  }

  mse /= N;

  free(hRefOutputs);

  return {mse};
}
// Usage
//  ./attention inputs.npy query.npy key.npy value.npy out_w.npy out_b.npy
//  output.npy <dim> <heads> <dim_head>
int main(int argc, char *argv[]) {
  if (argc != 13) {
    printf("[E] Incorrect number of arguments\n");
    printf(
        "[I] Usage ./attention inputs.npy query.npy key.npy value.npy "
        "out_w.npy out_b.npy output.npy <dim> <heads> <dim_head> <seq_len> "
        "<kernel_type> \n");
    return -1;
  }

  int dim, heads, dim_head, seq_len, kernel_type;

  dim = atoi(argv[8]);
  heads = atoi(argv[9]);
  dim_head = atoi(argv[10]);
  seq_len = atoi(argv[11]);
  kernel_type = atoi(argv[12]);

  printf(
      "[I] Loaded config: [dim: %d, heads: %d, dim_head %d, seq_len %d, "
      "kernel_type %d]\n",
      dim, heads, dim_head, seq_len, kernel_type);

  auto [hInputs, hQuery, hKey, hValue, hOutputWt, hOutputBias, hOutputs] =
      allocateAndLoadHostMemory(argv[1], argv[2], argv[3], argv[4], argv[5],
                                argv[6], argv[7], dim, heads, dim_head,
                                seq_len);

  auto [dInputs, dQuery, dKey, dValue, dOutputWt, dOutputBias, dOutputs] =
      allocateAndCopyDeviceMemory(hInputs, hQuery, hKey, hValue, hOutputWt,
                                  hOutputBias, hOutputs, dim, heads, dim_head,
                                  seq_len);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // Forward kernel
  for (int i = 0; i < 100; i++) {
    forward(dInputs, dQuery, dKey, dValue, dOutputWt, dOutputBias, dOutputs,
            dim, heads, dim_head, seq_len);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("[I] Kernel took : %fms\n", milliseconds / 100);

  auto [mse] = compareOutputs(hOutputs, dOutputs, dim * dim_head * heads);

  printf("[I] Mean squared error: %f\n", mse);
  
  deallocateHostMemory(hInputs, hQuery, hKey, hValue, hOutputWt, hOutputBias,
                       hOutputs);
  deallocateDeviceMemory(dInputs, dQuery, dKey, dValue, dOutputWt, dOutputBias,
                         dOutputs);

  printf("[I] Completed\n");
}
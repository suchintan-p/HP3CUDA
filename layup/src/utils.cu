#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <algorithm>
#include "helper_cuda.h"

// CUDA block width
#define BW 1024
#define EPSILON 1e-6


template<typename T> void cudaMemsetType(T *dev_ptr, T val, int n_vals)
{
    thrust::device_ptr<T> thrust_dev_ptr(dev_ptr);
    thrust::fill(thrust_dev_ptr, thrust_dev_ptr + n_vals, val);
}

float CrossEntropyLoss(float* pred_Y, float* true_Y, int n, int c, int h, int w)
{
    // Inialize loss on the device to be zero
    float loss, *d_loss;
    CUDA_CALL( cudaMalloc(&d_loss, sizeof(float)) );
    cudaMemsetType<float>(d_loss, 0.0, 1);

    // Accumulate the total loss on the device by invoking a kernel
    int n_blocks = std::min(65535, (n * c * h * w + BW  - 1) / BW);

	CrossEntropyKernel<<<n_blocks, BW, BW*sizeof(float)>>>(pred_Y, true_Y, d_loss, n, c, h, w);
	

    // Copy back the accumulated loss on the device back to the host
    CUDA_CALL( cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaFree(d_loss) );

    // Return the average loss
    return loss;
}


float SoftThresholdAccuracy(float* pred_Y, float* true_Y,
    int n, int c, int h, int w)
{
    // Initialize the accuracy on the device to be zero
    float acc, *d_acc;
    CUDA_CALL( cudaMalloc(&d_acc, sizeof(float)) );
    cudaMemsetType<float>(d_acc, 0.0, 1);

    // Accumulate the total loss on the device by invoking a kernel
    int n_blocks = std::min(65535, (n * c * h * w + BW - 1) / BW);
    SoftThresholdAccKernel<<<n_blocks, BW, BW * sizeof(float)>>>(pred_Y, true_Y,
        d_acc, n, c, h, w);

    // Copy back the accumulated accuracy on the device back to the host
    CUDA_CALL(cudaMemcpy(&acc, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_acc));

    // Return the average accuracy
    return acc / static_cast<float>(n);
}




__global__ void CrossEntropyKernel(float* pred_Y, float* true_Y, float *loss,
    int n, int c, int h, int w)
{
    extern __shared__ float shmem[];


	int tid = blockDim.x*blockIdx.x + threadIdx.x;
    const int local_tid = threadIdx.x;
    /*if(tid == 0)
    {
        printf("Printing.\n");
        for(int i=0; i<100; i++)
            printf("%f ", pred_Y[i]);
        printf("\n");
        for(int i=0; i<100; i++)
            printf("%f ", true_Y[i]);
        printf("\n");
    }*/

    shmem[local_tid] = 0.0;
    while (tid < (n*c*h*w) )
	{
        if(pred_Y[tid] == 0)
            pred_Y[tid] = EPSILON;
        shmem[local_tid] -= log(pred_Y[tid]) * true_Y[tid];
        tid += gridDim.x*blockDim.x; // Only necessary when the number of blocks > 65535
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s /= 2)
	{
        if (local_tid < s)
		{
            shmem[local_tid] += shmem[local_tid + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        // printf("Atomic add  : %f\n", shmem[0]);
        atomicAdd(loss, shmem[0] / static_cast<float>(n));
    }
}


__global__ void SoftThresholdAccKernel(float* pred_Y, float* true_Y, float* acc,
    int n, int c, int h, int w)
{
    extern __shared__ float shmem[];
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;


    shmem[tid] = 0.0;
    for (; idx < n; idx += blockDim.x * gridDim.x)
    {
        unsigned idx_cur = idx * c * h * w;

        unsigned argmax_pred = 0, argmax_true = 0;
        for (unsigned j = 0; j < c * h * w; ++j)
        {
            if (pred_Y[idx_cur + argmax_pred] < pred_Y[idx_cur + j])
                argmax_pred = j;

            if (true_Y[idx_cur + argmax_true] < true_Y[idx_cur + j])
                argmax_true = j;
        }

        if (argmax_pred == argmax_true)
            shmem[tid] += 1.0;
    }
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shmem[tid] += shmem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(acc, shmem[tid]);
}

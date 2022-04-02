// MAIN LAYER ABSTRACTION CLASS WHICH WE INHERIT IN model.cpp AND ITS HEADERS PRESENT IN layers.hpp
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

#include "layers.hpp"
#include "utils.cuh"
#include "helper_cuda.h"
using namespace std;


Layer::Layer(Layer *prev, cublasHandle_t cublasHandle,
    cudnnHandle_t cudnnHandle)
{
    this->prev = prev;
    this->cublasHandle = cublasHandle;
    this->cudnnHandle = cudnnHandle;

    CUDNN_CALL( cudnnCreateTensorDescriptor(&in_shape) );
    if (prev)
    {
        cudnnDataType_t dtype;
        int n, c, h, w, nStride, cStride, hStride, wStride;
        CUDNN_CALL( cudnnGetTensor4dDescriptor(prev->get_out_shape(), &dtype,
            &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride) );
        printf("Previous layer : (%d, %d, %d, %d)\n", n, c, h, w);
        CUDNN_CALL( cudnnSetTensor4dDescriptor(in_shape, CUDNN_TENSOR_NCHW,
            dtype, n, c, h, w) );
    }
    CUDNN_CALL( cudnnCreateTensorDescriptor(&out_shape) );
}

// Free the memory being used for internal batch and error representations.
Layer::~Layer()
{
    if (out_batch != in_batch)
        CUDA_CALL(cudaFree(out_batch));

    if (grad_out_batch != grad_in_batch)
        CUDA_CALL(cudaFree(grad_out_batch));

    if (weights)
        CUDA_CALL(cudaFree(weights));

    if (biases)
        CUDA_CALL(cudaFree(biases));

    if (grad_weights)
        CUDA_CALL(cudaFree(grad_weights));

    if (grad_biases)
        CUDA_CALL(cudaFree(grad_biases));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_shape));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_shape));
}


size_t Layer::get_workspace_size() const
{
    return 0;
}

void Layer::set_workspace(float *workspace, size_t workspace_size)
{
    this->workspace = workspace;
    this->workspace_size = workspace_size;
}

cudnnTensorDescriptor_t Layer::get_in_shape() const
{
    return in_shape;
}

cudnnTensorDescriptor_t Layer::get_out_shape() const
{
    return out_shape;
}

Layer *Layer::get_prev() const
{
    return this->prev;
}

float *Layer::get_output_fwd() const
{
    return out_batch;
}

float *Layer::get_input_fwd() const
{
    return in_batch;
}


float *Layer::get_input_bwd() const
{
    return grad_out_batch;
}


float Layer::get_loss()
{
    assert(false && "Non-loss layer has no loss.");
    return 0;
}

float Layer::get_accuracy()
{
    assert(false && "Non-loss layer does not support accuracy estimates.");
    return 0;
}


void Layer::allocate_buffers()
{
    if (prev)
    {
        this->in_batch = prev->get_output_fwd();
        this->grad_in_batch = prev->get_input_bwd();
    }

    // Get the shape of the output
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype,
        &n, &c, &h, &w, &n_stride, &c_stride, &h_stride, &w_stride) );

    // out_batch and grad_out_batch have the same shape as the output
    int out_size = n * c * h * w;
    CUDA_CALL( cudaMalloc(&out_batch, out_size * sizeof(float)) );
    CUDA_CALL( cudaMalloc(&grad_out_batch, out_size * sizeof(float)) );

    // Allocate buffers for the weights and biases (if there are any)
    if (n_weights > 0)
    {
        CUDA_CALL( cudaMalloc(&weights, n_weights * sizeof(float)) );
        CUDA_CALL( cudaMalloc(&grad_weights, n_weights * sizeof(float)) );
    }
    if (n_biases > 0)
    {
        CUDA_CALL( cudaMalloc(&biases, n_biases * sizeof(float)) );
        CUDA_CALL( cudaMalloc(&grad_biases, n_biases * sizeof(float)) );
    }
}


void Layer::init_weights_biases()
{
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL(cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
        &n_stride, &c_stride, &h_stride, &w_stride));

    curandGenerator_t gen;
    float minus_half = -0.5;
    float range = 2 / sqrt(static_cast<float>(c * h * w));
    std::random_device rd;
    CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, rd()) );

    if (weights)
    {
        CURAND_CALL( curandGenerateUniform(gen, weights, n_weights) );
        CUBLAS_CALL( cublasSaxpy(cublasHandle, n_weights,
            &minus_half, weights, 1, weights, 1) );
        CUBLAS_CALL( cublasSscal(cublasHandle, n_weights, &range, weights, 1) );
    }

    if (biases)
        cudaMemsetType<float>(biases, 0.0f, n_biases);

    CURAND_CALL( curandDestroyGenerator(gen) );
}

void Layer::freeUnnecessary()
{
    if (prev)
    {
        in_batch = NULL;
        grad_in_batch = NULL;
    }
    if(out_batch)
    {
        CUDA_CALL(cudaFree(out_batch));
        out_batch = NULL;
    }
    if(grad_out_batch)
    {
        CUDA_CALL(cudaFree(grad_out_batch));
        grad_out_batch = NULL;
    }
}

void Layer::allocateOutput()
{
    if (prev)
    {
        in_batch = prev->out_batch; // in_batch = prev->out_batch
        if(prev->prev && (prev->prev->is_ckpt == 0))
        {
            CUDA_CALL( cudaFree(prev->in_batch) );
            prev->in_batch = NULL;
            prev->prev->out_batch = NULL;
        }
    }
    CUDA_CALL( cudaMalloc(&out_batch, output_size * sizeof(float)) );

}

void Layer::allocateOutputBackward()
{
    if (prev)
    {
        in_batch = prev->out_batch; // in_batch = prev->out_batch
    }
    CUDA_CALL( cudaMalloc(&out_batch, output_size * sizeof(float)) );
}

void Layer::transferOutputToHost(cudaStream_t transfer_stream)
{
    CUDA_CALL( cudaMallocHost((void**) &( h_out_batchPinned ), output_size * sizeof(float) ) );
    CUDA_CALL( cudaMemcpyAsync(( h_out_batchPinned ), ( out_batch ), output_size * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream) );
}

void Layer::transferOutputToDevice(cudaStream_t transfer_stream)
{
    CUDA_CALL( cudaMalloc( &out_batch, (output_size)*sizeof(float) ) );
    CUDA_CALL( cudaMemcpyAsync(( out_batch ), ( h_out_batchPinned ), output_size * sizeof(float), cudaMemcpyHostToDevice, transfer_stream) );
}


void Layer::freeOutputMem()
{
    CUDA_CALL( cudaFree(out_batch) );
    out_batch = NULL;
}

void Layer::allocate_grad_out_batch()
{
    CUDA_CALL( cudaMalloc(&grad_out_batch, output_size*sizeof(float)) );
}

void Layer::allocate_grad_in_batch()
{
    CUDA_CALL( cudaMalloc(&grad_in_batch, input_size*sizeof(float)) );
}

void Layer::allocateGradients()
{
    allocate_grad_in_batch();
    prev->grad_out_batch = grad_in_batch;
}

void Layer::setInputPrevOutput()
{
    in_batch = prev->out_batch;
}


Input::Input(int n, int c, int h, int w,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(nullptr, cublasHandle, cudnnHandle)
{
   
    cudnnDataType_t dtype;
    int n_temp, c_temp, h_temp, w_temp;
    int nStride, cStride, hStride, wStride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, 
        &n_temp, &c_temp, &h_temp, &w_temp, &nStride, &cStride, &hStride, &wStride) ); // Get the last cuda descriptor. CHECK if this is necessary to add.
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW, 
        dtype, n, c, h, w) ); // Settting the parameters for descriptor.
    printf("Input Constructor, Batch size : %d, Channels : %d, Height :  %d, Wdith : %d\n", n,c,h,w);
    allocate_buffers();    
}

Input::~Input() = default;

void Input::forward_pass() {}

void Input::backward_pass(float learning_rate) {}



Dense::Dense(Layer *prev, int out_dim,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    // Get the input shape for the layer and flatten it if needed
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
        &n_stride, &c_stride, &h_stride, &w_stride) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(in_shape, CUDNN_TENSOR_NCHW,
        dtype, n, c * h * w, 1, 1) );

    // Initialize the output shape to be N out_size-dimensional vectors
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW, dtype,
        n, out_dim, 1, 1));

    // Initialize local shape parameters appropriately
    this->batch_size = n;
    this->in_size = c * h * w;
    this->out_size = out_dim;

    // The weights matrix is in_size by out_size, and there are out_size biases
    this->n_weights = in_size * out_size;
    this->n_biases = out_size;
    allocate_buffers();
    init_weights_biases();

    // Allocate a vector of all ones (filled with thrust::fill)
    CUDA_CALL( cudaMalloc(&onevec, batch_size * sizeof(float)) );
    cudaMemsetType<float>(onevec, 1.0f, batch_size);
}

Dense::~Dense()
{
    CUDA_CALL( cudaFree(onevec) );
}


void Dense::forward_pass()
{
    float one = 1.0, zero = 0.0;
		
	CUBLAS_CALL( cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
		out_size, batch_size, in_size, 
		&one, 
		weights, in_size,
		in_batch, in_size,
		&zero,
		out_batch, out_size) );
	

    // out_batch += bias * 1_vec^T (to distribute bias to all outputs in
    // this minibatch of data)
    CUBLAS_CALL( cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        out_size, batch_size, 1,
        &one,
        biases, out_size,
        onevec, batch_size,
        &one,
        out_batch, out_size) );
}


void Dense::backward_pass(float learning_rate)
{
    float one = 1.0, zero = 0.0;

    CUBLAS_CALL( cublasSgemm( 
				cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
				in_size, out_size, batch_size,
				&one, 
				in_batch, in_size,
				grad_out_batch, out_size,
				&zero,
				grad_weights, in_size	
				) );
    // grad_biases = grad_out_batch * 1_vec
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_N,
        out_size, batch_size,
        &one,
        grad_out_batch, out_size,
        onevec, 1,
        &zero,
        grad_biases, 1) );

    CUBLAS_CALL( cublasSgemm(
				cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				in_size, batch_size, out_size,
				&one,
				weights, in_size,
				grad_out_batch, out_size,
				&zero,
				grad_in_batch, in_size 	
				) );

    // Descend along the gradients of weights and biases using cublasSaxpy
    float eta = -learning_rate;

    // weights = weights + eta * grad_weights
    CUBLAS_CALL( cublasSaxpy(
				cublasHandle, in_size*out_size,
				&eta,
				grad_weights, 1,
				weights, 1
				) );


    // Tbiases = biases + eta * grad_biases
    CUBLAS_CALL( cublasSaxpy(
				cublasHandle, out_size,
				&eta,
				grad_biases, 1,
				biases, 1
				) );
	// cout<<"Inside backward pass of dense layer..."<<endl;
}


Activation::Activation(Layer *prev, cudnnActivationMode_t activationMode,
    double coef, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    CUDNN_CALL( cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
           &nStride, &cStride, &hStride, &wStride));



    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW,
            dtype, n, c, h, w) );
    // printf("Inside Activation init, Output Shape (%d, %d, %d, %d)\n", n, c, h, w);

    allocate_buffers();


    CUDNN_CALL( cudnnCreateActivationDescriptor(&activation_desc) ); 
    CUDNN_CALL( cudnnSetActivationDescriptor(activation_desc, activationMode, 
        CUDNN_PROPAGATE_NAN, coef)  );
}

Activation::~Activation()
{
	CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
}


void Activation::forward_pass()
{
    float one = 1.0, zero = 0.0;
	// cout<<"Inside forward pass of Activation layer"<<endl;

    CUDNN_CALL( cudnnActivationForward(
			cudnnHandle, activation_desc, 
			&one,
			in_shape, in_batch, 
			&zero,
			out_shape, out_batch 	
			) );
    
}


void Activation::backward_pass(float learning_rate)
{
    float one = 1.0, zero = 0.0;

    CUDNN_CALL( cudnnActivationBackward(
				cudnnHandle, activation_desc,
				&one, 
				out_shape, out_batch,	
				out_shape, grad_out_batch,
				in_shape, in_batch,
				&zero,
				in_shape, grad_in_batch	
				) );
}


Conv2D::Conv2D(Layer *prev, int n_kernels, int kernel_size, int stride, int padding,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(
            in_shape, &dtype,
            &n, &c, &h, &w, 
            &n_stride, &c_stride, &h_stride, &w_stride
            ));
    printf("Inside conv2D input shape : %d, %d, %d, %d\n", n, c, h, w);
    // Compute nubmer of weights and biases
    this->n_weights = n_kernels * c * kernel_size * kernel_size;
    this->n_biases = n_kernels;


    CUDNN_CALL( cudnnCreateFilterDescriptor(&filter_desc) );
    CUDNN_CALL( cudnnSetFilter4dDescriptor(
            filter_desc, dtype,
            CUDNN_TENSOR_NCHW,
            n_kernels, c, kernel_size, kernel_size
            ));
    printf("Created filter descriptor : %d %d %d %d\n", n_kernels, c, kernel_size, kernel_size);

    CUDNN_CALL( cudnnCreateTensorDescriptor(&bias_desc) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(
        bias_desc, 
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        1, n_kernels, 1, 1) );


    CUDNN_CALL( cudnnCreateConvolutionDescriptor(&conv_desc) );
    CUDNN_CALL( cudnnSetConvolution2dDescriptor(
            conv_desc, 
            padding, padding, // Same Padding
            stride, stride, // Stride
            1, 1, // Dilation
            CUDNN_CONVOLUTION, dtype
            ) );


    // Set output shape descriptor
    CUDNN_CALL( cudnnGetConvolution2dForwardOutputDim(conv_desc,
        in_shape, filter_desc, &n, &c, &h, &w) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w) );

    // Get convolution algorithms to use
    CUDNN_CALL( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
        in_shape, filter_desc, conv_desc, out_shape,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo) );
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
        in_shape, out_shape, conv_desc, filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algo));
    CUDNN_CALL( cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
        filter_desc, out_shape, conv_desc, in_shape,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_data_algo) );

    // Allocate all relevant buffers and initialize filters and biases
    allocate_buffers();
    init_weights_biases();
}

Conv2D::~Conv2D()
{
    CUDNN_CALL( cudnnDestroyTensorDescriptor(bias_desc) );

    CUDNN_CALL( cudnnDestroyConvolutionDescriptor(conv_desc) );
    CUDNN_CALL( cudnnDestroyFilterDescriptor(filter_desc) );
}

size_t Conv2D::get_workspace_size() const
{
    size_t acc = 0, tmp = 0;
    CUDNN_CALL( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
        in_shape, filter_desc, conv_desc, out_shape, fwd_algo, &tmp) );
    acc = std::max(acc, tmp);
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
        in_shape, out_shape, conv_desc, filter_desc, bwd_filter_algo, &tmp));
    acc = std::max(acc, tmp);
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
        filter_desc, out_shape, conv_desc, in_shape, bwd_data_algo, &tmp));
    acc = std::max(acc, tmp);
    return acc;
}


void Conv2D::forward_pass()
{
    float zero = 0, one = 1;

    CUDNN_CALL( cudnnConvolutionForward(
        cudnnHandle, 
        &one, 
        in_shape, in_batch,
        filter_desc, weights,
        conv_desc,
        fwd_algo, 
        workspace, workspace_size,
        &zero,
        out_shape, out_batch
        ) );

    CUDNN_CALL( cudnnAddTensor(cudnnHandle,
        &one, bias_desc, biases,
        &one, out_shape, out_batch) );
    // printf("Done Forward pass in Conv2D.\n");
}


void Conv2D::backward_pass(float learning_rate)
{
    float zero = 0, one = 1;
    // printf("Inside backward_pass of Conv2D\n");

    CUDNN_CALL( cudnnConvolutionBackwardFilter(
        cudnnHandle, 
        &one,
        in_shape, in_batch,
        out_shape, grad_out_batch,
        conv_desc, bwd_filter_algo,
        workspace, workspace_size,
        &zero,
        filter_desc, grad_weights
        ) );

    // Compute the gradient with respect to the biases
    CUDNN_CALL( cudnnConvolutionBackwardBias(cudnnHandle,
        &one, out_shape, grad_out_batch,
        &zero, bias_desc, grad_biases) );



    CUDNN_CALL(cudnnConvolutionBackwardData(
            cudnnHandle, 
            &one, 
            filter_desc, weights,
            out_shape, grad_out_batch,
            conv_desc, bwd_data_algo,
            workspace, workspace_size,
            &zero,
            in_shape, grad_in_batch
        ));


    // Descend along the gradients of the weights and biases using cublasSaxpy
    float eta = -learning_rate;
    
    CUBLAS_CALL(cublasSaxpy(
            cublasHandle, 
            n_weights,
            &eta, 
            grad_weights, 1,
            weights, 1
        ) );

    CUBLAS_CALL(cublasSaxpy(
            cublasHandle, n_biases,
            &eta, 
            grad_biases, 1,
            biases, 1
        ) );
}


Pool2D::Pool2D(Layer* prev, int stride, cudnnPoolingMode_t mode,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle)
{
    CUDNN_CALL( cudnnCreatePoolingDescriptor(&pooling_desc) );
    CUDNN_CALL( cudnnSetPooling2dDescriptor(
        pooling_desc, mode, 
        CUDNN_PROPAGATE_NAN,
        stride, stride, // Window size. Change this behavior.
        0, 0, // padding
        stride, stride // Stride
        ) );

    // Set output shape
    int n, c, h, w;
    CUDNN_CALL( cudnnGetPooling2dForwardOutputDim(pooling_desc, in_shape,
        &n, &c, &h, &w) );
    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, n, c, h, w) );

    // Allocate output buffer
    allocate_buffers();
}

Pool2D::~Pool2D()
{
    CUDNN_CALL( cudnnDestroyPoolingDescriptor(pooling_desc) );
}

void Pool2D::forward_pass()
{
    float zero = 0, one = 1;
    CUDNN_CALL(
        cudnnPoolingForward(
        cudnnHandle, 
        pooling_desc, &one, 
        in_shape, in_batch,
        &zero,
        out_shape, out_batch    
        ));
    // printf("Done forward passs in Pool2D\n");
}


void Pool2D::backward_pass(float learning_rate)
{
    float zero = 0, one = 1;
    // printf("Inside backward_pass of Pool2D\n");

    CUDNN_CALL(cudnnPoolingBackward(
            cudnnHandle, pooling_desc,
            &one,
            out_shape, out_batch,
            out_shape, grad_out_batch,
            in_shape, in_batch,
            &zero,
            in_shape, grad_in_batch
        ) );
}


Loss::Loss(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Layer(prev, cublasHandle, cudnnHandle) {}

Loss::~Loss() = default;



SoftmaxCrossEntropy::SoftmaxCrossEntropy(Layer *prev,
    cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle)
: Loss(prev, cublasHandle, cudnnHandle)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;


    CUDNN_CALL( cudnnGetTensor4dDescriptor(in_shape, &dtype, &n, &c, &h, &w,
            &nStride, &cStride, &hStride, &wStride) );


    CUDNN_CALL( cudnnSetTensor4dDescriptor(out_shape, CUDNN_TENSOR_NCHW,
            dtype, n, c, h, w) );

    allocate_buffers();
}

SoftmaxCrossEntropy::~SoftmaxCrossEntropy() = default;

void SoftmaxCrossEntropy::forward_pass()
{
    float one = 1.0, zero = 0.0;

    CUDNN_CALL( cudnnSoftmaxForward(
				cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&one,
				in_shape, in_batch,
				&zero,
				out_shape, out_batch		
				) );

}


void SoftmaxCrossEntropy::backward_pass(float lr)
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );
    int size = n * c * h * w;

    float minus_one = -1.0;

   	CUDA_CALL( cudaMemcpy( grad_in_batch, out_batch, size*sizeof(float),  cudaMemcpyDeviceToDevice) );

	CUBLAS_CALL( cublasSaxpy(
				cublasHandle, size,
				&minus_one,
				grad_out_batch, 1,
				grad_in_batch, 1
				) );	


    float scale = 1.0f / static_cast<float>(n);
    CUBLAS_CALL( cublasSscal(cublasHandle, size, &scale, grad_in_batch, 1) );
}


float SoftmaxCrossEntropy::get_loss()
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );

    loss = CrossEntropyLoss(out_batch, grad_out_batch, n, c, h, w);
    return loss;
}


float SoftmaxCrossEntropy::get_accuracy()
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    CUDNN_CALL(cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride));

    acc = SoftThresholdAccuracy(out_batch, grad_out_batch, n, c, h, w);
    return acc;
}

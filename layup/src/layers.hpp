// header file for layers.cpp
#pragma once

#include <cudnn.h>
#include <cublas_v2.h>


class Layer
{
public:
    Layer(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    virtual ~Layer();
    Layer *get_prev() const;

    virtual void forward_pass() = 0;


    virtual void backward_pass(float learning_rate) = 0;


    void allocate_buffers();
    void init_weights_biases();

    virtual float get_loss();
    virtual float get_accuracy();

    float *get_output_fwd() const;
    float *get_input_fwd() const;
    float *get_input_bwd() const;
    cudnnTensorDescriptor_t get_in_shape() const;
    cudnnTensorDescriptor_t get_out_shape() const;

    virtual size_t get_workspace_size() const;
    void set_workspace(float *workspace, size_t workspace_size);


    float threshold;

    void freeUnnecessary();
    int input_size, output_size, is_ckpt = 0;
    void allocateOutput();
    void allocateOutputBackward();
    void transferOutputToHost(cudaStream_t transfer_stream);
    void transferOutputToDevice(cudaStream_t transfer_stream);
    void freeOutputMem();
    void allocate_grad_out_batch();
    void allocate_grad_in_batch();
    void allocateGradients();
    void setInputPrevOutput();
    float *h_out_batchPinned = NULL;

protected:
    Layer *prev = NULL;
    cudnnTensorDescriptor_t in_shape;
    cudnnTensorDescriptor_t out_shape;
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;
    float *workspace = nullptr;
    size_t workspace_size = 0;
    float *in_batch = nullptr;
    float *out_batch = nullptr;
    float *grad_out_batch = nullptr;
    float *grad_in_batch = nullptr;



    float *weights = nullptr;
    float *biases = nullptr;
    float *grad_weights = nullptr;
    float *grad_biases = nullptr;
    int n_weights = 0;
    int n_biases = 0;
};


class Input : public Layer
{
public:
    Input(int n, int c, int h, int w,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Input();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;
};



class Dense : public Layer
{
public:
    Dense(Layer *prev, int out_dim,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Dense();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    int in_size;
    int out_size;
    int batch_size;
    float *onevec;
};


class Activation : public Layer
{
public:
    Activation(Layer *prev, cudnnActivationMode_t activationMode, double coef,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Activation();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    cudnnActivationDescriptor_t activation_desc;
};


class Conv2D : public Layer
{
public:
    Conv2D(Layer *prev, int n_kernels, int kernel_size, int stride, int padding,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Conv2D();
    size_t get_workspace_size() const override;
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
};


class Pool2D : public Layer
{
public:
    Pool2D(Layer *prev, int stride, cudnnPoolingMode_t mode,
        cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Pool2D();
    void forward_pass() override;
    void backward_pass(float learning_rate) override;

private:
    cudnnPoolingDescriptor_t pooling_desc;
};


class Loss : public Layer {
public:
    Loss(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~Loss();

protected:
    float loss = 0;
    float acc = 0;
};



class SoftmaxCrossEntropy : public Loss {
public:
    SoftmaxCrossEntropy(Layer *prev, cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle);
    ~SoftmaxCrossEntropy();
    void forward_pass() override;
    void backward_pass(float lr) override;
    float get_loss() override;
    float get_accuracy() override;
};

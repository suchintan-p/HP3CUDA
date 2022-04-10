// header files for model.cpp
#pragma once

#include <vector>
#include <cublas_v2.h>
#include <cudnn.h>
#include "layers.hpp"

typedef struct _result
{
    float loss;
    float acc;
    float *predictions;
} result;

class Model
{
public:
    Model(int n, int c, int h = 1, int w = 1);
    ~Model();

    void add(std::string layer_type, std::vector<int> shape = {});
    void init_workspace();

    void train(const float *train_X, float *train_Y, float lr,
               int num_examples, int n_epochs, int pre_allocate_gpu);

    void profile(const float *train_X, float *train_Y,
                 int transfer_every_layer);

    float *predict(const float *pred_X, int num_examples);
    result *evaluate(const float *eval_X, float *eval_Y, int num_examples);

    std::vector<int> checkpoints;

    std::vector<float *> cpu_memory;
    void cudaFreeUnnecessary();

private:
    void profile_on_batch(const float *batch_X, float *batch_Y, int transfer_every_layer);
    void train_on_batch(const float *batch_X, float *batch_Y, float lr);
    void train_on_batch_forward(const float *batch_X, float *batch_Y, float lr);
    void train_on_batch_backward(const float *batch_X, float *batch_Y, float lr, float *acc, float *loss);
    float *predict_on_batch(const float *batch_X);
    result *evaluate_on_batch(const float *batch_X, float *batch_Y);

    void copy_input_batch(const float *batch_X);
    void copy_output_batch(const float *batch_Y);

    int get_output_batch_size(Layer *layer) const;
    bool has_loss;
    int batch_size;
    int input_size = 0;
    std::vector<Layer *> *layers;
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;

    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    float *workspace = nullptr;

    size_t workspace_size = 0;
};

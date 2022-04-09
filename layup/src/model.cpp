// the main code where we implement algorithm 1(profile_on_batch) and algorithm 2(train_on_batch_forward and train_on_batch_backward)
#include <string>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include<iomanip>
#include<chrono>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "layers.hpp"
#include "model.hpp"
#include "helper_cuda.h"

#define ABS(x) (((x) >= 0) ? (x) : -(x))
#define PRINT_TIME 0


Model::Model(int n, int c, int h, int w) {
    this->has_loss = false;
    this->batch_size = n;
    this->input_size = c * h * w;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );
    CUDNN_CALL( cudnnCreate(&cudnnHandle) );


    CUDA_CALL( cudaStreamCreate(&compute_stream) );
    CUDA_CALL( cudaStreamCreate(&transfer_stream) );

    CUBLAS_CALL( cublasSetStream(cublasHandle, compute_stream) );
    CUDNN_CALL( cudnnSetStream(cudnnHandle, compute_stream) );

    this->layers = new std::vector<Layer *>;
    this->layers->push_back(new Input(n, c, h, w, cublasHandle, cudnnHandle));
    printf("Model init done.\n");
}

Model::~Model() {
    std::vector<Layer *>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        delete *it;
    delete this->layers;
    if (workspace)
        CUDA_CALL( cudaFree(workspace) );
    CUBLAS_CALL( cublasDestroy(cublasHandle) );
    CUDNN_CALL( cudnnDestroy(cudnnHandle) );
}


void Model::add(std::string layer, std::vector<int> shape)
{
    assert(!this->has_loss && "Cannot add any layers after a loss function.");

    Layer *last = layers->back();
    std::transform(layer.begin(), layer.end(), layer.begin(), ::tolower);

    /* ReLU activation */
    if (layer == "relu")
    {
        layers->push_back(
            new Activation(last, CUDNN_ACTIVATION_RELU, 0.0,
                cublasHandle, cudnnHandle));
    }

    /* tanh activation */
    else if (layer == "tanh")
    {
        layers->push_back(
            new Activation(last, CUDNN_ACTIVATION_TANH, 0.0,
                cublasHandle, cudnnHandle));
    }

    /* Loss layers must also update that the model has a loss function */
    else if (layer == "softmax crossentropy" || layer == "softmax cross-entropy")
    {
        layers->push_back(new SoftmaxCrossEntropy(last, cublasHandle, cudnnHandle));
        this->has_loss = true;
    }

    /* Dense layer */
    else if (layer == "dense")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive output shape for dense layer.");
        layers->push_back(new Dense(last, shape[0], cublasHandle, cudnnHandle));
    }

    /* Convolutional layer */
    else if (layer == "conv")
    {
        assert(shape[0] > 0 &&
            "Must specify positive number of knernels for conv layer.");
        assert(shape[1] > 0 &&
            "Must specify positive kernel dimension for conv layer.");
        assert(shape[2] > 0 &&
            "Must specify positive stride for conv layer.");

        int padding = (shape[1] - 1) / 2;
        if (shape.size() == 4)
            padding = shape[3];
        layers->push_back(
            new Conv2D(last, shape[0], shape[1], shape[2], padding,
                cublasHandle, cudnnHandle));
    }

    /* Max pooling layer */
    else if (layer == "max pool")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive pooling dimension.");
        layers->push_back(
            new Pool2D(last, shape[0], CUDNN_POOLING_MAX,
                cublasHandle, cudnnHandle) );
    }

    else if (layer == "avg pool" || layer == "average pool" ||
        layer == "mean pool")
    {
        assert(!shape.empty() && shape[0] > 0 &&
            "Must specify positive pooling dimension.");
        layers->push_back(
            new Pool2D(last, shape[0],
                CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                cublasHandle, cudnnHandle) );
    }


    else assert(false && "Invalid layer specification.");
}


void Model::init_workspace()
{
    assert(this->has_loss && "All layers of model must have been added!");

    // Get the largest workspace needed by any layer
    std::vector<Layer *>::iterator it;
    for (it = layers->begin(); it != layers->end(); ++it)
        workspace_size = std::max(workspace_size, (*it)->get_workspace_size());

    if (workspace_size > 0)
    {
        CUDA_CALL( cudaMalloc(&workspace, workspace_size) );
        for (it = layers->begin(); it != layers->end(); ++it)
            (*it)->set_workspace(workspace, workspace_size);
    }
}

void Model::profile(const float *train_X, float *train_Y, float lr, int n_examples,
    int n_epochs, int transfer_every_layer)
{
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());
    int n_batches = n_examples / batch_size;

    std:: cout << "Inside Profiler" << std:: endl;
    for (int i = 0; i < 1; ++i)
    {
      

        // Train on every complete batch
        for (long curr_batch = 0; curr_batch < 1; curr_batch++)
        {
            const float *curr_batch_X = train_X + curr_batch * in_size;
            float *curr_batch_Y = train_Y + curr_batch * out_size;
            profile_on_batch(curr_batch_X, curr_batch_Y, lr, transfer_every_layer);
			//printf("Okay Stop after callin train on batch\n");
	   		//exit(0);	
        }
    }
      std:: cout << "Exiting Profiler" << std:: endl;
}
void Model::train(const float *train_X, float *train_Y, float lr, int n_examples,
    int n_epochs, int pre_allocate_gpu)
{
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());
    int n_batches = n_examples / batch_size;

    for (int i = 0; i < n_epochs; ++i)
    {
        std::cout << "Epoch " << i + 1 << std::endl;
        std::cout << "--------------------------------------------------------------";
        std::cout << std::endl;

        float acc = 0;
        float loss = 0;
        float time_taken = 0;
        // Train on every complete batch
        for (long curr_batch = 0; curr_batch < n_batches; curr_batch++)
        {
            #if PRINT_TIME
                cudaEvent_t seq_start, seq_end;
                cudaEventCreate(&seq_start);
                cudaEventCreate(&seq_end);
                CUDA_CALL(cudaEventRecord(seq_start,0));
            #endif

            const float *curr_batch_X = train_X + curr_batch * in_size;
            float *curr_batch_Y = train_Y + curr_batch * out_size;
            if(pre_allocate_gpu)
            {
                train_on_batch(curr_batch_X, curr_batch_Y, lr); //original behvaviour.
                acc += this->layers->back()->get_accuracy();
                loss += this->layers->back()->get_loss();
            }
            else
            {
                train_on_batch_forward(curr_batch_X, curr_batch_Y, lr); 
                train_on_batch_backward(curr_batch_X, curr_batch_Y, lr, &acc, &loss); 
            }
            
            #if PRINT_TIME
                cudaDeviceSynchronize();
                cudaEventRecord(seq_end,0);
                cudaEventSynchronize(seq_end);
                float time_for_one_iter = 0.0;
                cudaEventElapsedTime(&time_for_one_iter, seq_start, seq_end);
                time_taken += time_for_one_iter;
                if(curr_batch % 10 == 0)
                    printf("Average time for one iteration: %f\n", time_taken/(curr_batch+1));
            #endif
        }
        std::cout << "Loss: " << loss / n_batches;
        std::cout << ",\tAccuracy: " << (100 * acc / n_batches);
        std::cout << std::endl << std::endl;
        sleep(10);
    }
}


float *Model::predict(const float *pred_X, int n_examples)
{
    // variables in which to get input and output shape
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());

    /* Allocate array for predictions */
    float *pred_Y = new float[out_size * n_examples / batch_size];

    /* Predict on every complete batch */
    int n_batches = n_examples / batch_size;
    for (int curr_batch = 0; curr_batch < n_batches; ++curr_batch) {
        float *curr = predict_on_batch(pred_X + curr_batch * in_size);
        CUDA_CALL( cudaMemcpy(pred_Y + curr_batch * out_size, curr,
            out_size * sizeof(float), cudaMemcpyDeviceToHost) );
    }

    return pred_Y;
}


result *Model::evaluate(const float *eval_X, float *eval_Y, int n_examples)
{
    int in_size = get_output_batch_size(layers->front());
    int out_size = get_output_batch_size(layers->back());
    int n_batches = n_examples / batch_size;

    // Allocate array for predictions
    result *ret = new result;
    ret->predictions = new float[out_size * n_batches];
    ret->acc = 0;
    ret->loss = 0;

    // Predict on every complete batch
    for (int curr_batch = 0; curr_batch < n_batches; ++curr_batch) {
        const float *curr_batch_X = eval_X + curr_batch * in_size;
        float *curr_batch_Y = eval_Y + curr_batch * out_size;
        result *curr = evaluate_on_batch(curr_batch_X, curr_batch_Y);

        // Copy results from batch into accumulator for full sample
        CUDA_CALL( cudaMemcpy(ret->predictions + curr_batch * out_size,
            curr->predictions, out_size * sizeof(float), cudaMemcpyDeviceToHost) );
        ret->acc = (ret->acc * curr_batch + curr->acc) / (curr_batch + 1u);
        ret->loss = (ret->loss * curr_batch + curr->loss) / (curr_batch + 1u);
        delete curr;
    }

    std::cout << "Validation" << std::endl;
    std::cout << "----------------------------------------------------";
    std::cout << std::endl << "Loss: " << ret->loss;
    std::cout << ",\tAccuracy: " << ret->acc << std::endl << std::endl;
    return ret;
}




void Model::profile_on_batch(const float *batch_X, float *batch_Y, float lr, int transfer_every_layer)
{
    assert(this->has_loss && "Cannot train without a loss function.");

    // Copy input and output minibatches into the model's buffers
    // copy_input_batch(batch_X);
    // copy_output_batch(batch_Y);

    cudaEvent_t seq_start, seq_end, tran_start, tran_end;
    cudaEventCreate(&seq_start);
    cudaEventCreate(&seq_end);
    cudaEventCreate(&tran_start);
    cudaEventCreate(&tran_end);

    double cumulative = 0.0;
    // Do a forward pass through every layer
    int layer_num = 0;
    std::vector<Layer *>::iterator it;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << prop.name << std::endl;
    double maxFLOPS = 9.3 * 1.0e+12;
    double utilRate = 1.0;
    double bandwidth = 732 * 1.0e+9;

    double constfact = 9300 / 732.0;

    for (it = this->layers->begin(); it != this->layers->end(); ++it, layer_num++)
    {
        auto out_shape = (*it)->get_out_shape();
        cudnnDataType_t dtype;
        int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
        CUDNN_CALL(cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
                &n_stride, &c_stride, &h_stride, &w_stride));

        float *t_output;
        (*it)->output_size = n*c*h*w; 
        // (*it)->allocateOutput();
        // if(it == this->layers->begin())
        // {
        //     copy_input_batch(batch_X);
        //     CUDA_CALL( cudaDeviceSynchronize() );
        // }
        // if((*it) == this->layers->back())
        // {
        //     (*it)->allocate_grad_out_batch();
        //     copy_output_batch(batch_Y);
        //     CUDA_CALL( cudaDeviceSynchronize() );
        // }
        CUDA_CALL(cudaEventRecord(seq_start,0));
        CUDA_CALL(cudaMalloc((float**)&t_output,n*c*h*w*sizeof(float)));	
        (*it)->forward_pass();
        cudaDeviceSynchronize();
        cudaEventRecord(seq_end,0);	
        cudaEventSynchronize(seq_end);
        float time_taken_compute = 0.0;
        cudaEventElapsedTime(&time_taken_compute, seq_start, seq_end);
        CUDA_CALL(cudaFree(t_output));
        cumulative += time_taken_compute;
        std:: cout << std::setprecision(15) << std::fixed << std::endl;
        std::cout << "Layer Number : " << layer_num << std::endl;
        std:: cout << "Time Taken compute = " << time_taken_compute << std:: endl;
        std:: cout << "Time Taken compute cumulative = " << cumulative << std:: endl;

        float *current_output = (*it)->get_output_fwd();
        float *temp_output = (float *)malloc(n*c*h*w*sizeof(float));
      
        // Storing output size
        if((*it)->get_prev())
        {
            (*it)->input_size = (*it)->get_prev()->output_size;
            printf("Input Size : %d\n", (*it)->input_size);
        }
        printf("Ouput Size : %d\n", (*it)->output_size);
      
        cudaEventRecord(tran_start,0);
        CUDA_CALL( cudaMemcpyAsync(temp_output, current_output,
        n*c*h*w*sizeof(float), cudaMemcpyDeviceToHost, 0));
        cudaDeviceSynchronize();

        cudaEventRecord(tran_end,0);	
        cudaEventSynchronize(tran_end);
        float time_taken_transfer = 0.0;
        cudaEventElapsedTime(&time_taken_transfer, tran_start, tran_end);

        std::cout << "Layer Number : " << layer_num << std::endl;
        std:: cout << "Time Taken transfer = " << time_taken_transfer << std:: endl;

        float thresh = time_taken_transfer/time_taken_compute;
        float thresh_cumulative = time_taken_transfer/cumulative;
        std:: cout << "Thresh = " << thresh << "\n";
        std:: cout << "Cumulative Thresh = " << thresh_cumulative << "\n";
        free(temp_output);
        if(!transfer_every_layer)
        {
            if(thresh_cumulative < 2.0 || layer_num==0)
            {
                (this)->checkpoints.push_back(layer_num);
                this->ckpt_pointers.push_back((*it));
                // (this)->cpu_memory.push_back(temp_output);
                cumulative = 0.0;
                (*it)->is_ckpt = 1;
                std::cout<<"CHECKPOINT AT LAYER "<<layer_num<<std::endl;
            }
        }
        else // Transfer every layer flag is on
        {
            (this)->checkpoints.push_back(layer_num);
            this->ckpt_pointers.push_back((*it));
            // (this)->cpu_memory.push_back(temp_output);
            cumulative = 0.0;
            (*it)->is_ckpt = 1;
            std::cout<<"CHECKPOINT AT LAYER "<<layer_num<<std::endl;
        }
    }
}

void Model::cudaFreeUnnecessary()
{
    // printf("Inside freeing mem.\n");
    std::vector<Layer *>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
    {
        (*it)->freeUnnecessary();
    }
    printf("Freed all input and output memory.\n");
}


void Model::train_on_batch(const float *batch_X, float *batch_Y, float lr)
{
    assert(this->has_loss && "Cannot train without a loss function.");

    // Copy input and output minibatches into the model's buffers
    copy_input_batch(batch_X);
    copy_output_batch(batch_Y);

    // Do a forward pass through every layer
    std::vector<Layer *>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        (*it)->forward_pass();
    // Do a backward pass through every layer
    std::vector<Layer *>::reverse_iterator rit;
    for (rit = this->layers->rbegin(); rit != this->layers->rend(); ++rit)
        (*rit)->backward_pass(lr);
}

void Model::train_on_batch_forward(const float *batch_X, float *batch_Y, float lr)
{
    assert(this->has_loss && "Cannot train without a loss function.");

    // Copy input and output minibatches into the model's buffers

    // Do a forward pass through every layer
    std::vector<Layer *>::iterator it;
    int checkpoint_num = 0, layer_num = 0;
    printf("\nCKPT 1");
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
    {
        printf("\nLayer Num: %d", layer_num);
        printf("\nCKPT 2");
        auto out_shape = (*it)->get_out_shape();
        cudnnDataType_t dtype;
        int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
        CUDNN_CALL(cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
                &n_stride, &c_stride, &h_stride, &w_stride));

        float *t_output;
        printf("\nCKPT 3");
        (*it)->output_size = n*c*h*w; 
        // printf("Layer start %d\n", layer_num);

        (*it)->allocateOutput(); // Simply allocate output buffer and free prev input (except if prev->prev is ckpt).
        /*if(layer_num != 0)
        {
            printf("Layer : %d; Out batch : %d, In batch : %d, Prev out batch : %d \n", layer_num, (*it)->get_output_fwd(), (*it)->get_input_fwd(), (*it)->get_prev()->get_output_fwd());
        }*/
        printf("\nCKPT 4");
        if( (*it)->is_ckpt )
        {
            // printf("checkpoints number : %d\n", checkpoint_num);
            // if(layer_num != checkpoints[0])
            CUDA_CALL( cudaStreamSynchronize(transfer_stream) );
            if(checkpoint_num > 1)
            {
                ckpt_pointers[checkpoint_num - 2]->freeOutputMem();
                (*layers)[checkpoints[checkpoint_num - 2] + 1]->setInputPrevOutput();
            }  
            if(checkpoint_num != 0)
            {
                ckpt_pointers[checkpoint_num - 1]->transferOutputToHost(transfer_stream);
            }
            checkpoint_num++;
            // (*it)->copyInputToHost(transfer_stream);
        }
        /*if(layer_num != 0)
        {
            printf("Layer : %d; Out batch : %d, In batch : %d, Prev out batch : %d \n", layer_num, (*it)->get_output_fwd(), (*it)->get_input_fwd(), (*it)->get_prev()->get_output_fwd());
        }*/

        if(it == this->layers->begin())
        {
            copy_input_batch(batch_X);
            CUDA_CALL( cudaDeviceSynchronize() );
        }
        if((*it) == this->layers->back())
        {
            (*it)->allocate_grad_out_batch();
            copy_output_batch(batch_Y);
            CUDA_CALL( cudaDeviceSynchronize() );
        }
        printf("\nCKPT 5");
        /*if(layer_num != 0)
        {
            printf("Layer : %d; Out batch : %d, In batch : %d, Prev out batch : %d \n", layer_num, (*it)->get_output_fwd(), (*it)->get_input_fwd(), (*it)->get_prev()->get_output_fwd());
        }*/

        CUDA_CALL(cudaMalloc((float**)&t_output,n*c*h*w*sizeof(float)));
        cudaDeviceSynchronize();
        (*it)->forward_pass();
        cudaDeviceSynchronize();
        printf("\nCKPT 6");
        CUDA_CALL(cudaFree(t_output));
        CUDA_CALL( cudaStreamSynchronize(compute_stream) );

        // printf("Layer done %d\n", layer_num);
        /*if(layer_num != 0)
        {
            printf("Layer : %d; Out batch : %d, In batch : %d, Prev out batch : %d \n", layer_num, (*it)->get_output_fwd(), (*it)->get_input_fwd(), (*it)->get_prev()->get_output_fwd());
        }*/
        layer_num++;
    }

    CUDA_CALL( cudaStreamSynchronize(transfer_stream) );
    CUDA_CALL( cudaStreamSynchronize(compute_stream) );
    // ckpt_pointers[ckpt_pointers.size() - 2]->freeOutputMem();

    // int temp = 0;
    // for (it = this->layers->begin(); it != this->layers->end(); ++it)
    // {
    //     // printf("Layer : %d; Out batch : %d, In batch : %d \n", temp, (*it)->get_output_fwd(), (*it)->get_input_fwd());
    //     temp++;
    // }
    // for(it = this->layers->begin(); it != this->layers->end(); ++it) // DUMMMYYYYYYY PLEASE REMOVE DURING BACKWARD.
    // {
    //     if((*it)->is_ckpt)
    //     {
    //         CUDA_CALL( cudaFreeHost((*it)->h_out_batchPinned) );
    //     }
    // }
    // Do a backward pass through every layer
    // std::vector<Layer *>::reverse_iterator rit;
    // for (rit = this->layers->rbegin(); rit != this->layers->rend(); ++rit)
        // (*rit)->backward_pass(lr);
}

void Model::train_on_batch_backward(const float *batch_X, float *batch_Y, float lr, float *acc, float *loss)
{
    std::vector<Layer *>::iterator rit;
    int ckpt_index = ckpt_pointers.size() - 1;
    int curr_ckpt = checkpoints[ckpt_index];
    int next_ckpt = (*layers).size() - 1;
    for(rit = this->layers->end() - 1; ; --rit)
    {
        // printf("Current ckpt %d; Next ckpt : %d\n", curr_ckpt, next_ckpt);
        // printf("Out batch : %d, In batch : %d \n", (*rit)->get_output_fwd(), (*rit)->get_input_fwd());

        if( (ckpt_index >= 1) && ( !ckpt_pointers[ckpt_index - 1]->get_output_fwd() ) )
        {
            ckpt_pointers[ckpt_index - 1]->transferOutputToDevice(transfer_stream);
            (*layers)[checkpoints[ckpt_index - 1] + 1]->setInputPrevOutput();
            // CUDA_CALL( cudaStreamSynchronize(transfer_stream) ); // Remove this
        }

        for(int j = curr_ckpt + 1; j < next_ckpt; j++)
        {
            if(j != next_ckpt) // Change this.
            {
                (*this->layers)[j]->allocateOutputBackward(); // in_batch = prev->out_batch, allocate memory for out_batch.
                (*layers)[j+1]->setInputPrevOutput();
            }
            // printf("Forward %d. Out batch : %d, In batch : %d; Prev out batch : %d \n", j, (*this->layers)[j]->get_output_fwd(), (*this->layers)[j]->get_input_fwd(), (*this->layers)[j]->get_prev()->get_output_fwd());
            (*this->layers)[j]->forward_pass();
            // printf("Forward pass\n");
        }

        for(int k = next_ckpt; k > curr_ckpt; k--)
        {
            // printf("backward %d. Out batch : %d, In batch : %d; Prev out batch : %d \n", k, (*this->layers)[k]->get_output_fwd(), (*this->layers)[k]->get_input_fwd(), (*this->layers)[k]->get_prev()->get_output_fwd());
            (*this->layers)[k]->allocateGradients(); // allocate mem to grad_out_batch and prev->grad_in_batch = grad_out_batch
            (*this->layers)[k]->backward_pass(lr);
        }
        // printf("Current ckpt %d; Next ckpt : %d. Done.\n", curr_ckpt, next_ckpt);
        CUDA_CALL( cudaStreamSynchronize(transfer_stream) );
        CUDA_CALL( cudaStreamSynchronize(compute_stream) );

        if(next_ckpt == (*layers).size() - 1 && (curr_ckpt != next_ckpt))
        {
            (*acc) += this->layers->back()->get_accuracy();
            (*loss) += this->layers->back()->get_loss();
        }

        for(int j = curr_ckpt + 1; j <= next_ckpt; j++)
        {
            (*this->layers)[j]->freeUnnecessary();
            if(j == next_ckpt)
                CUDA_CALL( cudaFreeHost(((*this->layers)[next_ckpt])->h_out_batchPinned) );
        }

        next_ckpt = curr_ckpt;
        ckpt_index--;
        curr_ckpt = checkpoints[ckpt_index];


        if(rit == this->layers->begin() || next_ckpt == 0)
            break;
    }

    (*layers)[0]->freeUnnecessary();
}



float *Model::predict_on_batch(const float *batch_X) {

    // Copy input (batch_X) into input layer and nothing into output layer
    copy_input_batch(batch_X);

    // Do a forward pass through every layer
    std::vector<Layer*>::iterator it;
    for (it = this->layers->begin(); it != this->layers->end(); ++it)
        (*it)->forward_pass();

    // The predictions are the output of the last layer
    return this->layers->back()->get_output_fwd();
}


result *Model::evaluate_on_batch(const float *batch_X, float *batch_Y) {
    assert(this->has_loss && "Cannot evaluate without a loss function.");

    // Making predictions does a forward pass
    result *ret = new result;
    ret->predictions = predict_on_batch(batch_X);

    copy_output_batch(batch_Y);
    ret->acc = this->layers->back()->get_accuracy();
    ret->loss = this->layers->back()->get_loss();
    return ret;
}


void Model::copy_input_batch(const float *batch_X)
{
    Layer *input = layers->front();
    int in_size = get_output_batch_size(input);
    CUDA_CALL(cudaMemcpyAsync(input->get_output_fwd(), batch_X,
        in_size * sizeof(float), cudaMemcpyHostToDevice) );
}


void Model::copy_output_batch(const float *batch_Y)
{
    Layer *loss = layers->back();
    int out_size = get_output_batch_size(loss);
    CUDA_CALL( cudaMemcpyAsync(loss->get_input_bwd(), batch_Y,
        out_size * sizeof(float), cudaMemcpyHostToDevice) );
}


int Model::get_output_batch_size(Layer *layer) const
{
    cudnnDataType_t dtype;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    cudnnTensorDescriptor_t out_shape = layer->get_out_shape();
    CUDNN_CALL( cudnnGetTensor4dDescriptor(out_shape, &dtype, &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride) );

    return n * c * h * w;
}
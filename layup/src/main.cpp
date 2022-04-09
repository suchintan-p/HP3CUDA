// main.cpp where our model is defined and it calls rest of the functions
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include "model.hpp"
#include "MNISTParser.h"

#define CONV 0
int main(int argc, char **argv)
{
    std::string activation = "relu";
    int pre_allocate_gpu = 0, transfer_every_layer = 0;

    // Directory in which training and testing data are stored (default is this)
    std::string dirname = "../../data";

    int batch_size = 10;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--dir") == 0 || strcmp(argv[i], "-d") == 0)
        {
            i++;
            if (i < argc)
                dirname = argv[i];
        }

        else if (strcmp(argv[i], "--act") == 0 || strcmp(argv[i], "-a") == 0)
        {
            i++;
            if (i < argc)
                activation = argv[i];
        }

        else if (strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0)
        {
            i++;
            if (i < argc)
                batch_size = atoi(argv[i]);
        }

        else if (strcmp(argv[i], "--pre_allocate_gpu") == 0 || strcmp(argv[i], "-pag") == 0)
        {
            pre_allocate_gpu = 1;
        }

        else if (strcmp(argv[i], "--transfer_every_layer") == 0 || strcmp(argv[i], "-tel") == 0)
        {
            transfer_every_layer = 1;
        }

    }

    if(transfer_every_layer && pre_allocate_gpu)
    {
        printf("Both flags are turned on -- (transfer_every_layer, pre_allocate_gpu). Please select one.\n");
        exit(0);
    }

    // Load training set
    int n_train, c, h, w, n_classes;
    float *train_X, *train_Y;
    LoadMNISTData(dirname + "/train-images.idx3-ubyte",
        dirname + "/train-labels.idx1-ubyte",
        n_train, c, h, w, n_classes, &train_X, &train_Y);
    std::cout << "Loaded training set." << std::endl;

    // Initialize a model to classify the MNIST dataset

    Model *model = new Model(batch_size, c, h, w);

#if VGG
    std::cout << "VGG Model selected" << std::endl;
    model->add("conv", { 64, 3, 1, 100 });
    model->add("max pool", { 2 });
    model->add("relu");
    model->add("conv", { 64, 3, 1 });
    model->add("relu");
    model->add("max pool", { 2 });
    model->add("conv", { 128, 3, 1 });
    model->add("relu");
    model->add("conv", { 128, 3, 1 });
    model->add("relu");
    model->add("max pool", { 2 });
    model->add("conv", { 256, 3, 1 });
    model->add("relu");
    model->add("conv", { 256, 3, 1 });
    model->add("relu");
    model->add("conv", { 256, 3, 1 });
    model->add("relu");
    model->add("max pool", { 2 });
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("max pool", { 2 });
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("conv", { 512, 3, 1 });
    model->add("relu");
    model->add("max pool", { 2 });

    model->add("dense", { 4096 });
    model->add("relu");
    model->add("dense", { 4096 });
    model->add("relu");
    model->add("dense", { n_classes });
    
    model->add("softmax crossentropy");
    model->init_workspace();
    

#else

#if CONV
    model->add("conv", { 20, 5, 1 });
    model->add("max pool", { 2 });
    model->add(activation);
    model->add("conv", { 5, 3, 1 });
    model->add("max pool", { 2 });
#else
    printf("Before adding dense layer\n");
    model->add("dense", { 200 });
    model->add(activation);
    model->add("dense", { 200 });
    model->add(activation);
    model->add("dense", { 200 });
    model->add(activation);
    model->add("dense", { 200 });
    model->add(activation);
    model->add("dense", { 200 });
    printf("After adding dense layer\n");
#endif
    printf("Before adding activation layer\n");
    model->add(activation);
    printf("After adding activation layer\n");
    model->add("dense", { n_classes });
    printf("After adding dense layer 2\n");
    model->add("softmax crossentropy");
    printf("After adding Cross entropy layer layer\n");
    model->init_workspace();

#endif

    // Train the model on the training set for 25 epochs
    std::cout << "Predicting on " << n_classes << " classes." << std::endl;
    model->profile(train_X, train_Y, 0.01f, n_train, 2, transfer_every_layer);
  
    // return 0;
    if(!pre_allocate_gpu)
    {
        printf("Freeing all the memory.\n");
        model->cudaFreeUnnecessary();
    }
    sleep(10);
    model->init_workspace();
    model->train(train_X, train_Y, 0.01f, n_train, 2, pre_allocate_gpu);

    // Load test set
    int n_test;
    // float *test_X, *test_Y;
    // LoadMNISTData(dirname + "/test-images.idx3-ubyte",
    //     dirname + "/test-labels.idx1-ubyte",
    //     n_test, c, h, w, n_classes, &test_X, &test_Y);
    // std::cout << "Loaded test set." << std::endl;

    // // Evaluate model on the test set
    // result *res = model->evaluate(test_X, test_Y, n_test);

    // Delete all dynamically allocated data
    // delete[] res->predictions;
    // delete res;
    delete model;
    delete[] train_X;
    delete[] train_Y;
    // delete[] test_X;
    // delete[] test_Y;

    return 0;
}

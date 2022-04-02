# Layup-HP3
This is the implementation of the [Layup](https://dl.acm.org/doi/10.1145/3357238) paper as part of our Semester Term Project for "High Performance Parallel Programming" for the class of Spring 2021. 

## File Structure

Ensure the following directory structure:

```bash
├── data/
├── layup/
    ├── bin/
    ├── obj/
    ├── src/
        ├── MNISTParser.h
        ├── helper_cuda.h
        ├── layers.cpp
        ├── layers.hpp
        ├── main.cpp 
        ├── model.cpp
        ├── model.hpp
        ├── tags
        ├── utils.cu
        ├── utils.cuh
    ├── tags       
├── stats/
├── README.md
```

## Compilation 

```
cd layup
make
```

## Running Instructions

### To run the VGG Model : 
```
bin/vggnet --dir ./../data -b 20
```

### To run a simple Conv-Net : 
```
bin/conv-neuralnet --dir ./../data -b 20
```

To run any of the baseline add the following flag : 
```
--pag   # To preallocate GPU Memory.
--tel   # To transfer every layer to CPU.
```
Note :  To remove the logging/printing of average iteration time, please set the flag to 0 in the following line https://github.com/ShahRutav/Layup-HP3/blob/db3290722479105069d2ba652194e8b4ecd71057/layup/src/model.cpp#L19

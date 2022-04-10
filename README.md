# Layup-HP3
Colab notebook: https://colab.research.google.com/drive/1h3tyGa_uc6ig4cPR-oIUGrOVVg99Yeo1?usp=sharing
This is the implementation of the [Layup](https://dl.acm.org/doi/10.1145/3357238) paper as part of our Semester Term Project for "High Performance Parallel Programming" for the class of Spring 2022. 

## Compilation 
```
cd layup
make
```
## Running Instructions
You can follow the same steps as given in the notebook. Alternatively you can run the following:

The following should be run inside "HP3CUDA/" directory.
### To run the VGG Model : 
```
 ./layup/bin/vggnet --dir data --act relu -b 100 
```
### To run a simple Conv-Net : 
```
./layup/bin/conv-neuralnet --dir data --act relu -b 100
```
### To run a simple Dense-Net : 
```
./layup/bin/dense-neuralnet --dir data --act relu -b 100
```
-b to specify batch size (default is 10)

--act to specify activation layer type (default is relu)

To run any of the baseline add the following flag : 
```
--pag   # To preallocate GPU Memory.
--tel   # Perform CPU_GPU transfer on every layer
```

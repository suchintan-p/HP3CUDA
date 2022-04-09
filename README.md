# Layup-HP3
Colab notebook: https://colab.research.google.com/drive/1h3tyGa_uc6ig4cPR-oIUGrOVVg99Yeo1?usp=sharing
This is the implementation of the [Layup](https://dl.acm.org/doi/10.1145/3357238) paper as part of our Semester Term Project for "High Performance Parallel Programming" for the class of Spring 2022. 

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
### To run a simple Dense-Net : 
```
bin/dense-neuralnet --dir ./../data -b 20
```
To run any of the baseline add the following flag : 
```
--pag   # To preallocate GPU Memory.
--tel   # Perform CPU_GPU transfer on every layer
```

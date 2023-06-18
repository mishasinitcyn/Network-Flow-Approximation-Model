# Network Flow Approximation Model

## Purpose
The purpose of this research project is to create a Machine Learning model capable of estimating optimal flow source-sink networks and their respective maximum flow values. This model may serve as an approximation to the [Ford-Fulkerson algorithm](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm).

## Dataset
The dataset is comprised of 10,000 5x5 source-sink network adjacency matrices and their corresponding optimized matrices computed using the Ford-Fulkerson algorithm.  Maximum flow values are also included but not currently used in any part of the model building, training, or evaluation.

## Model Building
- Model: Feedforward Neural Network with three linear layers and two hidden layers
- Loss Function: Mean Squared Error Loss
- Optimizer: Adaptive Moment Estimation (Adam) with a learning rate of 0.001
- Epochs: 100

## Model Training
The model is trained on 10,000 samples of network graphs with their optimized counterpart, both in adjacency matrix form, using an 80/20 train-test split.

## Model Evaluation
The model currently lacks extensive evaluation. Elementary tests have shown that the model can generate optimized network flow estimates with an optimal flow accuracy between 85-99%. The `Network Flow Approximation Model` notebook contains the resulting graphs and plots them using the networkx plot library. Both the raw and pre-processed (edges under 0.1 rounded to zero) ML-generated matrices are included.

## Future Improvements
- Experiment with model (more layers, different model)
- Increase the size of the graphs from 5x5
- Improve the adjacency matrix generating function to create graphs that contain more than one path from source to sink. This can be done by adding extra conditions in the bfs section of the Ford-Fulkerson algorithm
- Extensive model evaluation and benchmarking to compare with Ford-Fulkerson

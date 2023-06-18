# Network Flow Approximation Model

## Purpose
The purpose of this research project is to create a Machine Learning model capable of estimating optimal flow source-sink networks (in adjacency matrix form) and their respective maximum flow values. This model may serve as an approximation to the [Ford-Fulkerson algorithm](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm).

## Dataset
The dataset is comprised of 5x5 source-sink network adjacency matrices and their correspoding optimized matrices computed via the Ford-Fulkerson algorithm.  Maximum flow values are also included but not currently in any part of the model building, training, or evaluation.

## Model Building
- Model: Feedforward Neural Network with three linear layers and two hidden layers
- Loss Function: Mean Squared Error Loss
- Optimizer: Adaptive Moment Estimation (Adam) with a learning rate of 0.001
- Epochs: 100

## Model Evaluation
Having trained the model on 10000 samples with an 80/20 train-test split, the model can generate optimized network flow estimates with an optimal flow accuracy between 85-99%. The `Network Flow Approximation Model` notebook contains the resulting graphs and plots them using the networkx plot library. Both the raw and pre-processed (edges under 0.1) ML-generated matrices are included.

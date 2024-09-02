# Gliatron
A simple machine learning library for C

## About
This is a library for C aimed at providing an abstraction in order to allow for a much easier and simplistic way to implement neural networks and other things related machine learning.

## Features
 - Forward pass
 - Gradient descent
 - Neurons && layers
 - Activation functions && derivatives

## Build && use
NOTE:
This library depends on math.h, so it is necessary to add ```-lm``` to your build flags
You need to include the header file in your C program, and when compiling adding the c source file, as such:
```
gcc -lm your_neuralnetwork.c gliatron.c -o your_neuronnetwork
```

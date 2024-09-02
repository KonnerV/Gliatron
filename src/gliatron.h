#ifndef _GLIATRON_H_
#define _GLIATRON_H_
#endif

#include <stdint.h>
#include <stddef.h>

typedef struct {
    double b;
    double* w;
    uint16_t n_w;
} neuron_t;

/// Maths
// Activation functions
double nn_tanh(double x);
double nn_artanh(double x);
double nn_tanh_prime(double x);
// Matrices
double** matrix_dotproduct(double** m1, double** m2, size_t m1_rows, size_t m1_cols, size_t m2_rows, size_t m2_cols);
double* vecmatrix_to_array(double** m, size_t n_rows, size_t n_cols);
double** array_to_vecmatrix(double* a, size_t len);
double** transpose_matrix(double** m, size_t n_rows, size_t n_cols);
double** entrywise_product_matrix(double** m1, double** m2, size_t n_rows, size_t n_cols);

// Misc
double rnd_float(void);

// Init
void init_neuron(neuron_t* neuron, uint16_t n);
void init_layers(neuron_t** layer, uint16_t  n_neurons, uint16_t n_neurons_prev);

// NN algos
double** compute(size_t n_layers, uint8_t* n_neurons, double*** w_matrix, double** bias_matrix, double** x_matrix, size_t n_rows, size_t n_cols, double (*activation)(double));
void grad_desc(neuron_t*** nn, double*** w_m, double** b_m, double** act_m, uint8_t n_layers, uint8_t* n_neurons, double (*act_inv)(double), double (*act_prime)(double), double** x, size_t x_rows, double** y, double lr);


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "gliatron.h"

/// Maths
// Activation functions
double nn_tanh(double x) {
    if (x>340) {
        x=340;
    }
    else if (x<-340) {
        x=-340;
    }
    double y = ((exp(2*x)-1)/(exp(2*x)+1));
    return y;
}

double nn_artanh(double x) {
    double eps = 1e-5;
    if (x >= 1.f) {
        x = 1.f-eps;
    }
    else if (x <= -1.f) {
        x = -1.f+eps;
    }
    return (double) (log(((x+1)/(1-x)))/2);
}

double nn_tanh_prime(double x) {
    return (double) ((-4.f*exp(2*x))/((exp(2*x)+1)*(exp(2*x)+1)));
}

// Matrices
double** matrix_dotproduct(double** m1, double** m2, size_t m1_rows, size_t m1_cols, size_t m2_rows, size_t m2_cols) {
    if (m1_cols != m2_rows) {
        printf("err\n");
        return NULL;
    }

    double** res = (double**)calloc(m1_rows, sizeof(double*));
    for (size_t i=0;i<m1_rows;++i) {
        res[i] = (double*)calloc(m2_cols, sizeof(double));
    }

    for (size_t i=0;i<m1_rows;++i) {
        for (size_t j=0;j<m2_cols;++j) {
            for (size_t k=0;k<m2_rows;++k) {
                res[i][j] += m1[i][k] * m2[k][j];
            }
        }

    }
    return res;
}

double* vecmatrix_to_array(double** m, size_t n_rows, size_t n_cols) {
    double* res_arr = (double*)malloc((n_rows*n_cols)*sizeof(double));
    for (size_t i=0;i<n_rows;++i) {
        for (size_t j=0;j<n_cols;++j) {
            res_arr[i+j] = m[i][j];
        }
    }
    return res_arr;
}

double** array_to_vecmatrix(double* a, size_t len) {
    double** res_m = (double**)malloc(len*sizeof(double*));
    for (size_t i=0;i<len;++i) {
        res_m[i] = (double*)malloc(1*sizeof(double));
    }
    for (size_t i=0;i<len;++i) {
        res_m[i][0] = a[i];
    }
    return res_m;
}

double** transpose_matrix(double** m, size_t n_rows, size_t n_cols) {
    double** transposed = (double**)malloc(n_cols*sizeof(double*));
    for (size_t i=0;i<n_cols;++i) {
        transposed[i] = (double*)malloc(n_rows*sizeof(double));
    }

    for (int32_t i=(n_cols-1);i!=-1;--i) {
        for (size_t j=0;j<n_rows;++j) {
            transposed[i][j] = m[j][i];
        }
    }
    return transposed;
}

double** entrywise_product_matrix(double** m1, double** m2, size_t n_rows, size_t n_cols) {
    double** res = (double**)malloc(n_rows*sizeof(double*));
    for (size_t i=0;i<n_rows;++i) {
        res[i] = (double*)malloc(n_cols*sizeof(double));
    }
    for (size_t i=0;i<n_rows;++i) {
        for (size_t j=0;j<n_cols;++j) {
            res[i][j] = m1[i][j]*m2[i][j];
        }
    }
    return res;
}

// Misc
double rnd_float(void) {
    srand(time(NULL));
    return rand()/((double)RAND_MAX)*2-1;
}

// Init
void init_neuron(neuron_t* neuron, uint16_t n) {
    if ((neuron->w = (double*)malloc(n*sizeof(double))) == NULL) {
        exit(-1);
    }
    for (size_t i=0;i<n;++i) {
        neuron->w[i] = rnd_float();
    }
    neuron->b = rnd_float();
    neuron->n_w = n;
    return;
}

void init_layers(neuron_t** layer, uint16_t  n_neurons, uint16_t n_neurons_prev) {
    if (((*layer) = (neuron_t*)malloc(n_neurons*sizeof(neuron_t))) == NULL) {
        printf("Err alloc 2\n");
        exit(-1);
    }
    for (size_t i=0;i<n_neurons;++i) {
        init_neuron(&((*layer)[i]), n_neurons_prev);
    }
    return;
}

// NN algos
double** compute(size_t n_layers, uint8_t* n_neurons, double*** w_matrix, double** bias_matrix, double** x_matrix, size_t n_rows, size_t n_cols, double (*activation)(double)) {
    double** res = (double**)calloc(n_layers, sizeof(double*));
    double** coefficient_vec = malloc(n_rows*sizeof(double*));
    for (size_t i=0;i<n_rows;++i) {
        coefficient_vec[i] = malloc(1*sizeof(double));
    }
    memcpy(&coefficient_vec[0][0], &x_matrix[0][0], n_rows*1*sizeof(double));

    double** aux;
    size_t n_neurons_prev = n_rows;

    for (size_t i=0;i<n_layers;++i) {
        aux = coefficient_vec;
        coefficient_vec = matrix_dotproduct(w_matrix[i], aux, n_neurons[i], n_neurons_prev, n_neurons_prev, 1);
        for (size_t j=0;j<n_neurons_prev;++j) {
            free(aux[j]);
        }
        free(aux);

        res[i] = vecmatrix_to_array(coefficient_vec, n_neurons[i], 1);
        for (size_t j=0;j<n_neurons[i];++j) {
            res[i][j] += bias_matrix[i][j];
            res[i][j] = (*activation)(res[i][j]);
        }
        n_neurons_prev = n_neurons[i];
    }
    for (size_t i=0;i<n_neurons_prev;++i) {
        free(coefficient_vec[i]);
    }
    free(coefficient_vec);
    return res;
}

void grad_desc(neuron_t*** nn, double*** w_m, double** b_m, double** act_m, uint8_t n_layers, uint8_t* n_neurons, double (*act_inv)(double), double (*act_prime)(double), double** x, size_t x_rows, double** y, double lr, size_t n_samples) {
    double** delta;
    double** delta_new;
    double** x_transposed;
    double** w_transposed;
    double** act_prime_m;
    double** act_vec;
    double** act_vec_transposed;
    act_prime_m = array_to_vecmatrix(act_m[n_layers-1], n_neurons[n_layers-1]);
    double** grad_l = array_to_vecmatrix(act_m[n_layers-1], n_neurons[n_layers-1]);
    for (size_t i=0;i<n_neurons[n_layers-1];++i) {
        grad_l[i][0] -= y[i][0];
        act_prime_m[i][0] = (*act_prime)((*act_inv)(act_m[i][0]));
    }

    double** del_w;
    delta = entrywise_product_matrix(grad_l, act_prime_m, n_neurons[n_layers-1], 1);
    for (size_t i=0;i<n_neurons[n_layers-1];++i) {
        free(act_prime_m[i]);
        free(grad_l[i]);
    }
    free(act_prime_m);
    free(grad_l);

    for (int32_t i=(n_layers-1);i!=-1;--i) {
        switch (i==0) {
            case 1: {
                x_transposed = transpose_matrix(x, x_rows, 1);
                del_w = matrix_dotproduct(delta, x_transposed, n_neurons[i], 1, 1, x_rows);
                
                for (size_t j=0;j<x_rows;++j) {
                    free(x_transposed[j]);
                }
                free(x_transposed);
                for (size_t j=0;j<n_neurons[i];++j) {
                    for (size_t k=0;k<x_rows;++k) {
                        (*nn)[i][j].w[k] += del_w[j][k]*lr;
                    }
                    free(del_w[j]);
                }
                free(del_w);
                for (size_t j=0;j<n_neurons[i];++j) {
                    (*nn)[i][j].b += delta[j][0]*lr;
                }

                for (size_t j=0;j<n_neurons[i];++j) {
                    free(delta[j]);
                }
                free(delta);
                break;
            }
            default: {
                act_vec = array_to_vecmatrix(act_m[i-1], n_neurons[i-1]);
                act_vec_transposed = transpose_matrix(act_vec, n_neurons[i-1], 1);
                del_w = matrix_dotproduct(delta, act_vec_transposed, n_neurons[i], 1, 1, n_neurons[i-1]);

                for (size_t j=0;j<n_neurons[i-1];++j) {
                    free(act_vec[j]);
                }
                free(act_vec_transposed[0]);
                free(act_vec_transposed);
                free(act_vec);

                for (size_t j=0;j<n_neurons[i];++j) {
                    for (size_t k=0;k<n_neurons[i-1];++k) {
                        (*nn)[i][j].w[k] += del_w[j][k]*lr;
                    }
                    free(del_w[j]);
                }
                free(del_w);

                for (size_t j=0;j<n_neurons[i];++j) {
                    (*nn)[i][j].b += delta[j][0]*lr;
                }

                w_transposed = transpose_matrix(w_m[i], n_neurons[i], n_neurons[i-1]);
                delta_new = (matrix_dotproduct(w_transposed, delta, n_neurons[i-1], n_neurons[i], n_neurons[i], 1));

                for (size_t j=0;j<n_neurons[i-1];++j) {
                    free(w_transposed[j]);
                }
                free(w_transposed);

                for (size_t j=0;j<n_neurons[i];++j) {
                    free(delta[j]);
                }
                free(delta);

                act_prime_m = array_to_vecmatrix(act_m[i-1], n_neurons[i-1]);
                for (size_t j=0;j<n_neurons[i-1];++j) {
                    act_prime_m[j][0] = (*act_prime)((*act_inv)(act_prime_m[j][0]));
                }
                delta = entrywise_product_matrix(delta_new, act_prime_m, n_neurons[i-1], 1);

                for (size_t j=0;j<n_neurons[i-1];++j) {
                    free(act_prime_m[j]);
                    free(delta_new[j]);
                }
                free(act_prime_m);
                free(delta_new);
                break;
            }
        }
    }

    return;
}


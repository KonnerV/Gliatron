
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

typedef struct {
    double b;
    double* w;
    uint16_t n_w;
} neuron_t;

// Maths stuff

// Activation functions
double nn_tanh(double x) {
    return (double) ((exp(2*x)-1)/(exp(2*x)+1));
}

double nn_artanh(double x) {
    double eps = 1e-5;
    if (x <= -1.f) {
        x = -1.f+eps;
    }
    if (x >= 1.f) {
        x = 1.f-eps;
    }
    return (double) (log(((x+1)/(1-x)))/2);
}

double nn_tanh_prime(double x) {
    return (double) ((-4.f*exp(2*x))/((exp(2*x)+1)*(exp(2*x)+1)));
}

double rnd_float(void) {
    FILE* f = fopen("/dev/random", "r");
    uint8_t rand_byte = fgetc(f);
    fclose(f);
    return (double) rand_byte / (double) 255;
}

void init_neuron(neuron_t* neuron, uint16_t n) {
    if ((neuron->w = (double*)malloc(n*sizeof(double))) == NULL) {
        printf("Err alloc 3\n");
        exit(-1);
    }
    for (size_t i=0;i<n;++i) {
        neuron->w[i] = rnd_float();
        printf("w: %f\n", neuron->w[i]);
    }
    neuron->b = rnd_float();
    printf("b: %f\n", neuron->b);
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
        res[i] = (double*)malloc(n_cols*sizeof(float));
    }
    for (size_t i=0;i<n_rows;++i) {
        for (size_t j=0;j<n_cols;++j) {
            res[i][j] = m1[i][j]*m2[i][j];
        }
    }
    return res;
}

double** compute(size_t n_layers, uint8_t* n_neurons, double*** w_matrix, double** bias_matrix, double** x_matrix, size_t n_rows, size_t n_cols, double (*activation)(double)) {
    double** res = (double**)malloc(n_layers*sizeof(double*));
    for (size_t i=0;i<n_layers;++i) {
        res[i] = (double*)calloc(n_neurons[i], sizeof(double));
    }

    double** coefficient_vec = x_matrix;
    size_t n_neurons_prev = n_rows;

    for (size_t i=0;i<n_layers;++i) {
        coefficient_vec = matrix_dotproduct(w_matrix[i], coefficient_vec, n_neurons[i], n_neurons_prev, n_neurons_prev, 1);
        res[i] = vecmatrix_to_array(coefficient_vec, n_neurons[i], 1);
        for (size_t j=0;j<n_neurons[i];++j) {
            res[i][j] += bias_matrix[i][j];
            res[i][j] = (*activation)(res[i][j]);
        }
        n_neurons_prev = n_neurons[i];
    }

    return res;
}

void grad_desc(neuron_t*** nn, double*** w_m, double** b_m, double** act_m, uint8_t n_layers, uint8_t* n_neurons, double (*act_inv)(double), double (*act_prime)(double), double** x, size_t x_rows, double** y, double lr) {
    double** delta;
    double** delta_new;
    double** act_prime_m = array_to_vecmatrix(act_m[n_layers-1], n_neurons[n_layers-1]);
    double** grad_l = array_to_vecmatrix(act_m[n_layers-1], n_neurons[n_layers-1]);
    for (size_t i=0;i<n_neurons[n_layers-1];++i) {
        grad_l[i][0] -= y[i][0];
        act_prime_m[i][0] = (*act_prime)((*act_inv)(act_prime_m[i][0]));
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
                del_w = matrix_dotproduct(delta, transpose_matrix(x, x_rows, 1), n_neurons[i], 1, 1, x_rows);
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
                del_w = matrix_dotproduct(delta, transpose_matrix(array_to_vecmatrix(act_m[i-1], n_neurons[i]), n_neurons[i], 1), n_neurons[i], 1, 1, n_neurons[i]);
                for (size_t j=0;j<n_neurons[i];++j) {
                    for (size_t k=0;k<n_neurons[i];++k) {
                        (*nn)[i][j].w[k] += del_w[j][k]*lr;
                    }
                    free(del_w[j]);
                }
                free(del_w);

                for (size_t j=0;j<n_neurons[i];++j) {
                    (*nn)[i][j].b += delta[j][0]*lr;
                }

                delta_new = (matrix_dotproduct(transpose_matrix(w_m[i], n_neurons[i], n_neurons[i-1]), delta, n_neurons[i-1], n_neurons[i], n_neurons[i], 1));
                // freeing the old delta after it's use
                for (size_t j=0;j<n_neurons[i];++j) {
                    free(delta[j]);
                }
                free(delta);

                act_prime_m = array_to_vecmatrix(act_m[i-1], n_neurons[i-1]);
                for (size_t j=0;j<n_neurons[i-1];++j) {
                    act_prime_m[j][0] = (*act_prime)((*act_inv)(act_prime_m[j][0]));
                }
                delta = entrywise_product_matrix(delta_new, act_prime_m, n_neurons[i-1], 1);

                // freeing "new_delta" so we can create the delta for the previous layer && freeing the activation prime matrix as we no longer need it
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


int main(int argc, char** argv) {
    neuron_t** nn;
    uint8_t n_layers = 1;
    uint8_t n_neurons[1] = {1};
    if ((nn = (neuron_t**)malloc(n_layers*sizeof(neuron_t*))) == NULL) {
        printf("Err alloc 1\n");
        exit(-1);
    }
    int8_t n_neurons_prev = 2;
    for (size_t i=0;i<n_layers;++i) {
        init_layers(&nn[i], n_neurons[i], n_neurons_prev);
        n_neurons_prev = n_neurons[i];
    }
    // TODO: Tidy up main
    double*** w_m = (double***)malloc(n_layers*sizeof(double**));
    double** b_m = (double**)malloc(n_layers*sizeof(double*));
    for (size_t i=0;i<n_layers;++i) {
        w_m[i] = (double**)malloc(n_neurons[i]*sizeof(double*));
        b_m[i] = (double*)malloc(n_neurons[i]*sizeof(double));
        for (size_t j=0;j<n_neurons[i];++j) {
            w_m[i][j] = (double*)malloc(nn[i][j].n_w*sizeof(double));
        }
    }

    double* input_m[2] = {
        (double[]) {0.3f},
        (double[]) {0.5f},
    };

    double* y_m[1] = {
        (double[]) {0.f}
    };

    double** act_m;
    size_t epochs = 5;
    for (size_t c_epoch=0;c_epoch<epochs;++c_epoch) {
        for (size_t i=0;i<n_layers;++i) {
            for (size_t j=0;j<n_neurons[i];++j) {
                for (size_t k=0;k<nn[i][j].n_w;++k) {
                    w_m[i][j][k] = nn[i][j].w[k];
                }
                b_m[i][j] = nn[i][j].b;
            }
        }

        act_m = compute(n_layers, n_neurons, w_m, b_m, input_m, 1, 1, &nn_tanh);
        grad_desc(&nn, w_m, b_m, act_m, n_layers, n_neurons, &nn_artanh, &nn_tanh_prime, input_m, 2, y_m, 0.3);
        for (size_t i=0;i<n_layers;++i) {
            printf("[");
            for (size_t j=0;j<n_neurons[i];++j) {
                printf("%f,", act_m[i][j]);
            }
            printf("]\n");
        }

    }

    for (size_t i=0;i<n_layers;++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            free(w_m[i][j]);
        }
        free(w_m[i]);
        free(b_m[i]);
    }
    free(w_m);
    free(b_m);

    for (size_t i=0;i<n_layers;++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            free(nn[i][j].w);
        }
        free(nn[i]);
    }
    free(nn);
    for (size_t i=0;i<n_layers;++i) {
        free(act_m[i]);
    }
    free(act_m);
    return 0;
}

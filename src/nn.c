
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

typedef struct {
    float b;
    float* w;
    uint16_t n_w;
} neuron_t;

// Maths stuff

// Activation functions
float nn_tanh(float x) {
    return (float) ((expf(2*x)-1)/(expf(2*x)+1));
}

float nn_artanh(float x) {
    return (float) (logf(((x+1)/(1-x)))/2);
}

float nn_tanh_prime(float x) {
    return (float) ((-4.f*expf(2*x))/((expf(2*x)+1)*(expf(2*x)+1)));
}

float softplus(float x) {
    return (float) logf(1+expf(x));
}

float softplus_prime(float x) {
    return (float) (1/(1+expf(-x)));
}

float softplus_inv(float x) {
    return (float) logf(expf(x)-1);
}

// loss functions
float mse(float predictions, float actual) {
    float res = 0;
    //for (size_t i=0;i<n_samples;++i) {
    //    res += (actual[i]-predictions[i])*(actual[i]-predictions[i]);
    //}
    res = ((actual-predictions)*(actual-predictions));
    //res /= (float) n_samples;
    return res;
}

float mse_prime(float predictions, float actual) {
    float res = 0;
    //res += 2*(x-y);
    //for (size_t i=0;i<n_samples;++i) {
    //    res += 2*(x-y);
    //}
    //res /= (float)n_samples;
    res = 2*(predictions-actual);//predictions-actual;

    return res;
}

float rnd_float(void) {
    FILE* f = fopen("/dev/random", "r");
    uint8_t rand_byte = fgetc(f);
    fclose(f);
    return (float) rand_byte / (float) 255;
}

void init_neuron(neuron_t* neuron, uint16_t n) {
    if ((neuron->w = (float*)malloc(n*sizeof(float))) == NULL) {
        printf("Err alloc 3\n");
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
        printf("w: %f, b: %f\n", (*layer)[i].w[0], (*layer)[i].b);
    }
    return;
}


float** matrix_multiply(float** m1, float** m2, size_t m1_rows, size_t m1_cols, size_t m2_rows, size_t m2_cols) {
    float** res = (float**)calloc(m1_rows, sizeof(float*));
    for (size_t i=0;i<m1_cols;++i) {
        res[i] = (float*)calloc(m2_cols, sizeof(float));
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

float** transpose_matrix(float** m, size_t n_rows, size_t n_cols) {
    float** transposed = (float**)calloc(n_cols, sizeof(float*));
    for (size_t i=0;i<n_cols;++i) {
        transposed[i] = (float*)calloc(n_rows, sizeof(float));
    }

    for (int32_t i=(n_cols-1);i!=-1;--i) {
        for (size_t j=0;j<n_rows;++j) {
            transposed[i][j] = m[j][i];
        }
    }
    return transposed;
}

float** compute(size_t n_layers, uint8_t* n_neurons, float*** w_matrix, float** bias_matrix, float** input_matrix, size_t n_rows, size_t n_cols, float (*activation)(float)) {
    float** res = (float**)malloc(n_layers*sizeof(float*));
    for (size_t i=0;i<n_layers;++i) {
        res[i] = (float*)calloc(n_neurons[i], sizeof(float));
    }
    // we may be able to determine n_neurons_prev within this function
    float** coefficient_vec = input_matrix;
    size_t n_neurons_prev = n_rows;

    for (size_t i=0;i<n_layers;++i) {
        res[i] = *matrix_multiply(w_matrix[i], coefficient_vec, n_neurons[i], n_neurons_prev, n_rows, n_cols);
        for (size_t j=0;j<n_neurons[i];++j) {
            res[i][j] += bias_matrix[i][j];
            res[i][j] = (*activation)(res[i][j]);
        }
        coefficient_vec = &(res[i]);
        n_neurons_prev = n_neurons[i];
    }

    return res;
}

int main(int argc, char** argv) {
    neuron_t** nn;
    uint8_t n_layers = 5;
    uint8_t n_neurons[5] = {1, 1, 1, 1, 1};
    if ((nn = (neuron_t**)malloc(n_layers*sizeof(neuron_t*))) == NULL) {
        printf("Err alloc 1\n");
        exit(-1);
    }
    int8_t n_neurons_prev = 1;
    for (size_t i=0;i<n_layers;++i) {
        init_layers(&nn[i], n_neurons[i], n_neurons_prev);
        n_neurons_prev = n_neurons[i];
    }
    // TODO: Tidy up main
    float*** w_m = (float***)malloc(n_layers*sizeof(float**));
    float** b_m = (float**)malloc(n_layers*sizeof(float*));
    for (size_t i=0;i<n_layers;++i) {
        w_m[i] = (float**)malloc(n_neurons[i]*sizeof(float*));
        b_m[i] = (float*)malloc(n_neurons[i]*sizeof(float));
        for (size_t j=0;j<n_neurons[i];++j) {
            w_m[i][j] = (float*)malloc(nn[i][j].n_w);
        }
    }

    for (size_t i=0;i<n_layers;++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            for (size_t k=0;k<nn[i][j].n_w;++k) {
                w_m[i][j][k] = nn[i][j].w[k];
            }
            b_m[i][j] = nn[i][j].b;
        }
    }


    float* input_m[1] = {
        (float[]) {2.f}
    };
    float** act_m = compute(n_layers, n_neurons, w_m, b_m, input_m, 1, 1, &softplus);
    for (size_t i=0;i<n_layers;++i) {
        printf("[");
        for (size_t j=0;j<n_neurons[i];++j) {
            printf("%f,", act_m[i][j]);
        }
        printf("]\n");
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

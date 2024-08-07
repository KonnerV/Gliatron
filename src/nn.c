
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>

#define ARR_LEN(x) sizeof(x)/sizeof(x[0])

typedef struct {
    int x;
    int y;
} train_data;

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

float leaky_ReLU(float x) {
    if (x>0) {
        return x;
    }
    return (float) 0.01*x;
}

float leaky_ReLU_prime(float x) {
    if (x>0) {
        return (float) 1;
    }
    return (float) 0.01;
}

float softplus(float x) {
    return (float) logf(1+expf(x));
}

float softplus_prime(float x) {
    return (float) (1/(1+expf(-x)));
}

// loss functions
float mse(neuron_t neuron, train_data* data) {
    float res = 0;
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += ((data[i].x*neuron.w[0]-data[i].y)*(data[i].x*neuron.w[0]-data[i].y));
    }
    res /= (float) ARR_LEN(data);
    return res;
}

float mse_prime(float x, float y, uint8_t n_samples) {
    float res = 0;
    /*
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += (((2*data[i].x*data[i].x)*neuron.w[0])-(2*data[i].x*data[i].y));
    }
    res /= (float)ARR_LEN(data);
    */
    // TODO: work on proper implementation
    res = (2*(x-y));///(float)n_samples;

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

float** compute(neuron_t** nn, uint8_t n_layers, uint8_t* n_neurons, float (*activation)(float), float x) {
    float** y = (float**)malloc(n_layers*sizeof(float*));
    for (size_t i=0;i<n_layers;++i) {
        y[i] = (float*)malloc(n_neurons[i]*sizeof(float));
    }

    float a=x;
    float z = 0;
    for (size_t i=0;i<(n_layers-1);++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            for (size_t k=0;k<nn[i][j].n_w;++k) {
                z += nn[i][j].w[k] * a;    
            }
            z += nn[i][j].b;
            y[i][j] = (*activation)(z);
        }
        a = (*activation)(z);
        z=0;
    }
    for (int i=0;i<n_neurons[n_layers-1];++i) {
        for (int j=0;j<nn[n_layers-1][i].n_w;++j) {
            z += nn[n_layers-1][i].w[j]*a;
        }
        z += nn[n_layers-1][i].b;
        y[n_layers-1][i] = (*activation)(z);
    }
    return y;
}

void learn(neuron_t*** nn, float** act_matrix, uint8_t n_layers, uint8_t* n_neurons, float (*act_inv)(float), float (*act_prime)(float), float (*l_prime)(float, float, uint8_t), float y) {
    float del_w = 0;
    float del_b = 0;
    float del_a = (*l_prime)((act_matrix[n_layers-1][n_neurons[n_layers-1]-1]), y, 1);
    float a_prevk = 0;
    float a_cur = 0;
    for (size_t i=(n_layers-1);i!=-1;--i) {
        for (size_t j=(n_neurons[i]-1);j!=-1;--j) {
            for (size_t k=0;k<n_neurons[i];++k) {
                a_prevk = act_matrix[i][k];
                a_cur = act_matrix[i][j];
                del_w = a_prevk*((*act_prime)(((*act_inv)(a_cur))))*del_a;
                del_a *= (*nn)[i][j].w[k]*((*act_prime)(((*act_inv)(a_cur))));
                printf("del_w:%f, del_a:%f\n", del_w, del_a);
            }
        }
    }

    return;
}

int main(int argc, char** argv) {
    neuron_t** nn;
    uint8_t n_layers = 2;
    uint8_t n_neurons[2] = {1, 1};
    if ((nn = (neuron_t**)malloc(2*sizeof(neuron_t))) == NULL) {
        printf("Err alloc 1\n");
        exit(-1);
    }
    for (size_t i=0;i<n_layers;++i) {
        init_layers(&nn[i], 1, 1);
    }

    // Tidy up main
    float** act_m = compute(nn, n_layers, n_neurons, &nn_tanh, 2.f);
    for (size_t i=0;i<n_layers;++i) {
        printf("[");
        for (size_t j=0;j<n_neurons[i];++j) {
            printf("%f,", act_m[i][j]);
        }
        printf("]\n");
    }
    learn(&nn, act_m, n_layers, n_neurons, &nn_artanh, &nn_tanh_prime, &mse_prime, 4.f);

    for (size_t j=0;j<n_layers;++j) {
        for (size_t k=0;k<1;++k) {
            free(nn[j][k].w);
        }
        free(nn[j]);
    }
    free(nn);
    return 0;
}

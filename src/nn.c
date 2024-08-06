
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

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

// loss functions
float mse(neuron_t neuron, train_data* data) {
    float res = 0;
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += ((data[i].x*neuron.w[0]-data[i].y)*(data[i].x*neuron.w[0]-data[i].y));
    }
    res /= (float) ARR_LEN(data);
    return res;
}

float mse_prime(float x, float y) {
    float res = 0;
    /*
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += (((2*data[i].x*data[i].x)*neuron.w[0])-(2*data[i].x*data[i].y));
    }
    res /= (float)ARR_LEN(data);
    */
    // TODO: work on proper implementation
    res = 2*(x-y);

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

float* compute(neuron_t** nn, uint8_t n_layers, uint8_t* n_neurons, uint8_t n_last_layer, float (*activation)(float), float x) {
    float* y = (float*)malloc(n_last_layer*sizeof(float));
    float a=x;
    float z = 0;
    for (size_t i=0;i<(n_layers-1);++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            for (size_t k;k<nn[i][j].n_w;++k) {
                z += nn[i][j].w[k] * a;    
            }
            z += nn[i][j].b;
        }
        a = (*activation)(z);
    }
    for (int i=0;i<n_neurons[n_layers-1];++i) {
        for (int j=0;j<nn[n_layers-1][i].n_w;++j) {
            z += nn[n_layers-1][i].w[j]*a;
        }
        z += nn[n_layers-1][i].b;
        y[i] = (*activation)(z);
    }
    return y;
}

void calc_del(neuron_t*** nn, uint8_t n_layers, uint8_t* n_neurons, float x, float y, float (*activation)(float), float (*activation_prime)(float), float (*loss_prime)(float, float)) {
    float a = x;
    float z;
    float del_w = 0;
    float del_b = 0;

    for (size_t i=0;i<n_layers;++i) {
        for (size_t j=0;j<n_neurons[i];++j) {
            z = 0;
            for (size_t k=0;k<(*nn)[i][j].n_w;++k) {
                z += (*nn)[i][j].w[k]*a;
            }
            z += (*nn)[i][j].b;
            float v = (*activation)(z);
            del_w = a*((*activation_prime)(z))*((*loss_prime)(v, y));
            del_b = 1*((*activation_prime)(z))*((*loss_prime)(v, y));
            (*nn)[i][j].b += del_b;
            for (size_t k=0;k<(*nn)[i][j].n_w;++k) {
                (*nn)[i][j].w[k] += del_w;
            }
            a = (*activation)(z);
        }

    }

    return;
}

void train(neuron_t** nn, uint8_t n_layers, uint8_t* n_neurons, size_t epochs, float (*loss)(float, float)) {
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

    printf("Computed y: %f\n", *compute(nn, n_layers, n_neurons, 1, &nn_tanh, 2.f));

    for (size_t j=0;j<n_layers;++j) {
        for (size_t k=0;k<1;++k) {
            free(nn[j][k].w);
        }
        free(nn[j]);
    }
    free(nn);
    return 0;
}


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

float nn_tanh_prime(float x) {
    return (float) ((-4.f*expf(2*x))/((expf(2*x)+1)*(expf(2*x)+1)));
}

// loss functions

float loss(neuron_t neuron, train_data* data) {
    float res = 0;
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += ((data[i].x*neuron.w[0]-data[i].y)*(data[i].x*neuron.w[0]-data[i].y));
    }
    res /= (float) ARR_LEN(data);
    return res;
}

float loss_prime(neuron_t neuron, train_data* data) {
    float res = 0;
    for (size_t i=0;i<ARR_LEN(data);++i) {
        res += (((2*data[i].x*data[i].x)*neuron.w[0])-(2*data[i].x*data[i].y));
    }
    res /= (float)ARR_LEN(data);
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
    if ((*layer = (neuron_t*)malloc(n_neurons*sizeof(neuron_t))) == NULL) {
        printf("Err alloc 2\n");
        exit(-1);
    }
    for (size_t i=0;i<n_neurons;++i) {
        init_neuron(&((*layer)[i]), n_neurons_prev);
        printf("w: %f, b: %f\n", (*layer)[i].w[0], (*layer)[i].b);
    }
    return;
}

float compute(neuron_t** nn, uint8_t n_layers, uint8_t n_neurons, float x) {
    float y=0;
    for (size_t j=0;j<n_layers;++j) {
        for (size_t k=0;k<n_neurons;++k) {
            for (size_t l;l<nn[j][k].n_w;++l) {
                y += nn[j][k].w[l] * x;    
            }
            y += nn[j][k].b;
        }
        y = nn_tanh(y);
    }
    return y;
}

int main(int argc, char** argv) {
    neuron_t** nn;
    uint8_t n_layers = 2;
    uint8_t n_neurons = 1;
    if ((nn = (neuron_t**)malloc(2*sizeof(neuron_t*))) == NULL) {
        printf("Err alloc 1\n");
        exit(-1);
    }
    for (size_t i=0;i<n_layers;++i) {
        init_layers(&nn[i], n_neurons, n_neurons);
    }

    printf("Computed y: %f\n", compute(nn, n_layers, n_neurons, 2.f));

    for (size_t j=0;j<n_layers;++j) {
        for (size_t k=0;k<n_neurons;++k) {
            free(nn[j][k].w);
        }
        free(nn[j]);
    }
    free(nn);
    return 0;
}

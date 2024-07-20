
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define ARR_LEN(x) sizeof(x)/sizeof(x[0])

// training data type, just to make data manipulation easier
typedef struct {
    int x;
    int y;
} train_data;


// Each Neuron has a bias, weight(s), and an activation function
typedef struct {
    float b;
    float* w;
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
    if ((neuron->w = (float*)calloc(n, sizeof(float))) == NULL) {
        printf("Err alloc 3\n");
        exit(-1);
    }
    for (size_t i=0;i<n;++i) {
        neuron->w[i] = rnd_float();
    }
    neuron->b = rnd_float();
    return;
}

void init_layers(neuron_t** layer, uint16_t  n_neurons) {
    if ((*layer = (neuron_t*)calloc(n_neurons, sizeof(neuron_t))) == NULL) {
        printf("Err alloc 2\n");
        exit(-1);
    }
    for (size_t i=0;i<n_neurons;++i) {
        init_neuron(&((*layer)[i]), 1);
        printf("w: %f, b: %f\n", (*layer)[i].w[0], (*layer)[i].b);
    }
    return;
}

int main(int argc, char** argv) {
    neuron_t** nn;
    uint8_t n_layers = 2;
    uint8_t n_neurons = 1;
    if ((nn = (neuron_t**)calloc(2, sizeof(neuron_t*))) == NULL) {
        printf("Err alloc 1\n");
        exit(-1);
    }
    for (size_t i=0;i<n_layers;++i) {
        init_layers(&nn[i], n_neurons);
    }


    for (size_t j=0;j<n_layers;++j) {
        for (size_t k=0;k<n_neurons;++k) {
            free(nn[j][k].w);
        }
        free(nn[j]);
    }
    free(nn);
    return 0;
}

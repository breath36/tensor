
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
    TENSOR_FLOAT32,
    TENSOR_FLOAT16,
    TENSOR_INT8
} TensorType;

typedef struct {
    TensorType type;
    uint16_t length;
    float scale;
    int8_t zero_point;
    union {
        float    *f32;
        uint16_t *f16;
        int8_t   *i8;
    } data;
} Tensor;

uint16_t float_to_half(float f) {
    uint32_t x = *((uint32_t*)&f);
    uint16_t sign = (x >> 16) & 0x8000;
    uint32_t mantissa = x & 0x7FFFFF;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mantissa >> 13);
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) f = sign;
    else {
        exp = exp - 15 + 127;
        f = sign | (exp << 23) | (mant << 13);
    }
    return *((float*)&f);
}

Tensor tensor_create(TensorType type, uint16_t length) {
    Tensor t;
    t.type = type;
    t.length = length;
    t.scale = 1.0f;
    t.zero_point = 0;
    switch (type) {
        case TENSOR_FLOAT32: t.data.f32 = (float*)malloc(sizeof(float) * length); break;
        case TENSOR_FLOAT16: t.data.f16 = (uint16_t*)malloc(sizeof(uint16_t) * length); break;
        case TENSOR_INT8:    t.data.i8 = (int8_t*)malloc(sizeof(int8_t) * length); break;
    }
    return t;
}

void tensor_set(Tensor *t, uint16_t idx, float value) {
    if (idx >= t->length) return;
    switch (t->type) {
        case TENSOR_FLOAT32: t->data.f32[idx] = value; break;
        case TENSOR_FLOAT16: t->data.f16[idx] = float_to_half(value); break;
        case TENSOR_INT8:    t->data.i8[idx] = (int8_t)(value / t->scale) + t->zero_point; break;
    }
}

float tensor_get(Tensor *t, uint16_t idx) {
    if (idx >= t->length) return 0.0f;
    switch (t->type) {
        case TENSOR_FLOAT32: return t->data.f32[idx];
        case TENSOR_FLOAT16: return half_to_float(t->data.f16[idx]);
        case TENSOR_INT8:    return (t->data.i8[idx] - t->zero_point) * t->scale;
    }
    return 0.0f;
}

float dense_forward(Tensor *weights, Tensor *input, Tensor *bias) {
    float acc = 0.0f;
    uint16_t i;
    for (i = 0; i < input->length; i++) {
        acc += tensor_get(weights, i) * tensor_get(input, i);
    }
    return acc + tensor_get(bias, 0);
}

int main() {
    printf("--- Gomulu AI Sunum Demosu ---\n\n");

    Tensor input = tensor_create(TENSOR_INT8, 1);
    input.scale = 0.1f; 
    tensor_set(&input, 0, 22.4f); 

    Tensor weight = tensor_create(TENSOR_FLOAT16, 1);
    tensor_set(&weight, 0, -1.2f); 

    Tensor bias = tensor_create(TENSOR_FLOAT32, 1);
    tensor_set(&bias, 0, 50.0f);

    float sonuc = dense_forward(&weight, &input, &bias);

    printf("Girdi Sicakligi: 22.40 C\n");
    printf("Bellekteki INT8 Karsiligi: %d (Quantized)\n", input.data.i8[0]);
    printf("Hesaplanan Isitici Gucu: %%%.2f\n", sonuc);

    free(input.data.i8); 
    free(weight.data.f16); 
    free(bias.data.f32);

    printf("\nDevam etmek icin bir tusa basin...");
    getchar(); 
    return 0;
}

#pragma once

void attention_cuda(
    float* Q,
    float* K,
    float* V,
    float* O,
    int N,
    int d
);
#ifndef UTILS_H
#define UTILS_H

#include <cstddef>

float* read_binary(
    const char* filename,
    size_t size
);

void write_binary(
    const char* filename,
    float* data,
    size_t size
);

#endif // UTILS_H
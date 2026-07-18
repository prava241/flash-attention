#include <fstream>
#include <cstddef>

#include "utils.h"

float* read_binary(
    const char* filename,
    size_t size
)
{
    float* data = new float[size];

    std::ifstream file(filename, std::ios::binary);

    file.read(
        reinterpret_cast<char*>(data),
        size * sizeof(float)
    );

    file.close();

    return data;
}

void write_binary(
    const char* filename,
    float* data,
    size_t size
)
{
    std::ofstream file(
        filename,
        std::ios::binary
    );

    file.write(
        reinterpret_cast<char*>(data),
        size*sizeof(float)
    );

    file.close();
}
#pragma once
#include "tensor.h"
#include <memory>

class Pooling {
public:
    Pooling(std::size_t kernel_size, std::size_t stride);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

private:
    int _kernel_size;
    int _stride;
};

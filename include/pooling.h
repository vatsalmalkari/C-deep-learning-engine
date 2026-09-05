// pooling.h

#pragma once
#include "tensor.h"
#include <memory>
#include <vector>

class Pooling {
private:
    std::size_t _kernel_size;
    std::size_t _stride;
    Arena* _arena;

public:
    Pooling(std::size_t kernel_size, std::size_t stride = 2, Arena* arena = nullptr);
    
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
};

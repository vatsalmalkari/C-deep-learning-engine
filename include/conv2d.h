#pragma once

#include <vector>
#include <string>
#include <memory>
#include "tensor.h"
#include "arena_wrapper.h"

class Conv2D {
private:
    std::size_t _in_channels;
    std::size_t _out_channels;
    std::size_t _kernel_size;
    std::size_t _stride;
    std::size_t _padding;
    ArenaAllocator* _allocator;
    std::size_t _seed;

    // Smart pointers to manage Tensor lifetime cleanly
    std::shared_ptr<Tensor> _weight;
    std::shared_ptr<Tensor> _bias;

    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> _parameters;

    void register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor);

public:
    // Default argument values specified HERE in the declaration:
    Conv2D(std::size_t in_channels, std::size_t out_channels, 
           std::size_t kernel_size, std::size_t stride = 1, 
           std::size_t padding = 0, ArenaAllocator* allocator = nullptr, 
           std::size_t seed = 42);

    ~Conv2D();

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters();
};
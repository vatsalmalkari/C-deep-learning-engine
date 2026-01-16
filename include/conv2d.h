#pragma once
#include "module.h"
#include "tensor.h"
#include <memory>

class Conv2D : public Module
{
public:
    Conv2D(std::size_t in_channels,
           std::size_t out_channels,
           std::size_t kernel_size,
           std::size_t stride = 1,
           std::size_t padding = 0,
           std::size_t seed = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

private:
    std::size_t _in_channels;
    std::size_t _out_channels;
    std::size_t _kernel_size;
    std::size_t _stride;
    std::size_t _padding;
    std::size_t _seed;

    std::shared_ptr<Tensor> _weight;  // [C_out, C_in, K, K]
    std::shared_ptr<Tensor> _bias;    // [C_out]
};

#pragma once
#include "module.h"
#include "tensor.h"

class Relu : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
#pragma once
#include "module.h"
#include "tensor.h"
#include <memory>

class Softmax : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
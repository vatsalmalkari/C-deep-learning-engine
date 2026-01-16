#pragma once
#include "module.h"
#include "tensor.h"
#include <memory>

class Flatten : public Module
{
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
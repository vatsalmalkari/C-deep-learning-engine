#ifndef DROPOUT_H
#define DROPOUT_H

#include "module.h"
#include "../include/tensor.h"
#include <vector>

class Dropout : public Module {
    float rate;
    bool is_training;
    std::shared_ptr<Tensor> mask; 

public:
   
    Dropout(float rate = 0.5f);

    void train() { is_training = true; }
    void eval() { is_training = false; }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() {
        return {}; 
    }
};

#endif 
#include "../include/dropout.h"
#include "../include/tensor.h"
#include <random>
#include <algorithm>
#include <vector>

Dropout::Dropout(float rate) : rate(rate), is_training(true) {}

std::shared_ptr<Tensor> Dropout::forward(std::shared_ptr<Tensor> input) {
    // Pass through directly
    if (!is_training) {
        return input;
    }

    //Generate Mask
    float scale = 1.0f / (1.0f - rate);
    
    static std::mt19937 gen(1234); 
    std::bernoulli_distribution d(1.0f - rate);

    const auto& in_data = input->data(); 
    std::vector<float> out_data;
    out_data.reserve(in_data.size());
    
    std::vector<float> mask_vec; 
    mask_vec.reserve(in_data.size());

    //  Forward Pass
    for (float val : in_data) {
        if (d(gen)) {
            out_data.push_back(val * scale);
            mask_vec.push_back(scale);
        } else {
            out_data.push_back(0.0f);
            mask_vec.push_back(0.0f);
        }
    }

    // Gradient Function
    auto grad_fn = [input, mask_vec](const std::vector<float>& grad_output) {
        
        std::vector<float> grad_input;
        grad_input.reserve(grad_output.size());

        // Chain Rule: d(Input) = d(Output) * Mask
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input.push_back(grad_output[i] * mask_vec[i]);
        }

      
        input->add_to_grad(grad_input);
    };
    
    auto result = std::make_shared<Tensor>(
        out_data,             
        input->shape(),        
        input->requires_grad(),
        grad_fn,             
        std::vector<std::shared_ptr<Tensor>>{input} 
    );

    return result;
}
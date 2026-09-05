#include "../include/dropout.h"
#include "../include/tensor.h"
#include <algorithm>
#include <vector>
#include <cstdlib>

Dropout::Dropout(float rate) : rate(rate), is_training(true) {}

std::shared_ptr<Tensor> Dropout::forward(std::shared_ptr<Tensor> input) {
    // Pass through directly
    if (!is_training) {
        return input;
    }

    // Generate Mask
    float scale = 1.0f / (1.0f - rate);
    
    // Seed rand() once statically
    static bool seed_once = []() {
        std::srand(1234); 
        return true;
    }();

    // Calculate probability threshold
    float p = 1.0f - rate;

    const float* in_data = input->data();
    std::size_t numel = input->size();
    
    std::vector<float> out_data(numel);
    std::vector<float> mask_vec(numel);

    // Forward Pass - evaluate probability per element
    for (std::size_t i = 0; i < numel; ++i) {
        bool keep = ((float)std::rand() / (float)RAND_MAX) < p;
        
        if (keep) {
            out_data[i] = in_data[i] * scale;
            mask_vec[i] = scale;
        } else {
            out_data[i] = 0.0f;
            mask_vec[i] = 0.0f;
        }
    }

    // Gradient Function
    auto grad_fn = [input, mask_vec](const std::vector<float>& grad_output) {
        std::vector<float> grad_input(grad_output.size());
        for (std::size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = grad_output[i] * mask_vec[i];
        }
        input->add_to_grad(grad_input);
    };
    
    // Renamed this to output_tensor to avoid naming conflict
    auto output_tensor = std::make_shared<Tensor>(
        out_data,             
        input->shape(),        
        input->requires_grad(),
        grad_fn,             
        std::vector<std::shared_ptr<Tensor>>{input} 
    );

    return output_tensor;
}

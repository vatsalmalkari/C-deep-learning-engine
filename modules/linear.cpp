#include "../include/linear.h"
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <stdexcept>

Linear::Linear(std::size_t in_features, std::size_t out_features, std::size_t seed)
    : _in_features(in_features), _out_features(out_features)
{
    std::vector<float> w(_in_features * _out_features);
    
    float limit = sqrt(6.0f / (_in_features + _out_features));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-limit, limit);
    
    for (size_t i = 0; i < w.size(); i++) {
        w[i] = distribution(generator);
    }
    
    std::vector<float> b(_out_features, 0.0f);

    _weight = std::make_shared<Tensor>(w, std::vector<std::size_t>{_out_features, _in_features}, true);
    _bias = std::make_shared<Tensor>(b, std::vector<std::size_t>{_out_features}, true);

    register_parameter("weight", _weight);
    register_parameter("bias", _bias);
}

void Linear::reset_parameters() {}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input)
{
    // Input is flattened 
    if (input->numel() != _in_features) {
         throw std::runtime_error("Linear input size mismatch. Expected " + 
             std::to_string(_in_features) + " but got " + std::to_string(input->numel()));
    }

    // Prepare Output
    std::vector<float> out(_out_features);
    const auto& in_data = input->data();
    const auto& w_data = _weight->data();
    const auto& b_data = _bias->data();

    // Forward Pass: Y = W * X + B. W as [Out, In] to dot-product the rows
    for (size_t i = 0; i < _out_features; i++) {
        float sum = b_data[i];
        for (size_t j = 0; j < _in_features; j++) {
            
            sum += w_data[i * _in_features + j] * in_data[j];
        }
        out[i] = sum;
    }

    // Backward Pass
    if (input->requires_grad()) {

        std::vector<std::shared_ptr<Tensor>> parents = {input, _weight, _bias};
        
        // gradients by value/shared_ptr
        std::function<void(const std::vector<float>&)> gradfn = [input, weight=_weight, bias=_bias, in_f=_in_features, out_f=_out_features]
            (const std::vector<float>& grad_output) 
        {
            std::vector<float> grad_input(in_f, 0.0f);
            std::vector<float> grad_weight(weight->numel(), 0.0f);
          
            std::vector<float> grad_bias = grad_output; 
            
            const auto& in_vals = input->data();
            const auto& w_vals = weight->data();

            for (size_t i = 0; i < out_f; i++) {
                float g = grad_output[i];
                
                for (size_t j = 0; j < in_f; j++) {
                    // dL/dW_ij = input_j * grad_output_i
                    grad_weight[i * in_f + j] = in_vals[j] * g;

                    // dL/dX_j += W_ij * grad_output_i
                    // This sends the gradient back to the CNN!
                    grad_input[j] += w_vals[i * in_f + j] * g;
                }
            }
            
            // Push gradients to parents
            input->add_to_grad(grad_input);
            weight->add_to_grad(grad_weight);
            bias->add_to_grad(grad_bias);
        };

        return std::make_shared<Tensor>(out, std::vector<std::size_t>{_out_features}, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(out, std::vector<std::size_t>{_out_features});
}
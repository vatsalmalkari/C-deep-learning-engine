#include "../include/flatten.h"
#include "../include/tensor.h"
#include <functional>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input)
{
    // The data is flat in memory, regardless of whether shape is 2D, 3D, or 4D.
    const std::vector<float>& in_data = input->data();
    
    // output data
    std::vector<float> out_data = in_data;

    // Output Shape: 
    std::vector<std::size_t> out_shape = { in_data.size() };

    // Handle Gradient
    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        // gradient for flatten is just a pass-through. The numbers don't change shape interpretation
        std::function<void(const std::vector<float> &)> gradfn = 
            [input](const std::vector<float> &grad_output)
        {
            // Direct pass-through of gradients
            input->add_to_grad(grad_output);
        };
        
        return std::make_shared<Tensor>(out_data, out_shape, true, gradfn, parents);
    }

    return std::make_shared<Tensor>(out_data, out_shape);
}
#include "../include/relu.h"
#include "../include/tensor.h"
#include <functional>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input)
{
    // Access raw flat data (works for 1D, 2D, 3D, 4D)
    const std::vector<float>& in_data = input->data();
    std::vector<float> out_data;
    out_data.reserve(in_data.size());

    // ReLU element-wise
    for (const float& val : in_data)
    {
        out_data.push_back(val > 0.0f ? val : 0.0f);
    }

    // Gradient
    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        std::function<void(const std::vector<float> &)> gradfn =
            [input](const std::vector<float> &grad_output)
        {
            const std::vector<float>& input_vals = input->data();
            std::vector<float> grad_input;
            grad_input.reserve(grad_output.size());

            // Gradient is 1 if input > 0, else 0
            for (std::size_t i = 0; i < input_vals.size(); i++)
            {
                if (input_vals[i] > 0.0f)
                {
                    grad_input.push_back(grad_output[i]);
                }
                else
                {
                    grad_input.push_back(0.0f);
                }
            }
            input->add_to_grad(grad_input);
        };
        
        return std::make_shared<Tensor>(out_data, input->shape(), true, gradfn, parents);
    }

    return std::make_shared<Tensor>(out_data, input->shape());
}
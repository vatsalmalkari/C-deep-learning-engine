#include "../include/flatten.h"
#include "../include/tensor.h"
#include <functional>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input)
{
    std::size_t in_size = input->numel();
    std::vector<std::size_t> out_shape = { in_size };

    //a reference/pointer vector without element-by-element deep copies if possible.
    const float* in_data = input->data();
    std::vector<float> out_data(in_data, in_data + in_size); 

    bool req_grad = input->requires_grad();

    if (req_grad)
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        std::function<void(const std::vector<float> &)> gradfn = 
            [input](const std::vector<float> &grad_output)
        {
            // Direct structural pass-through of gradients
            input->add_to_grad(grad_output);
        };
        
        return std::make_shared<Tensor>(out_data, out_shape, true, gradfn, parents);
    }

    return std::make_shared<Tensor>(out_data, out_shape);
}

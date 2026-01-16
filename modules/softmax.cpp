#include "../include/softmax.h"
#include "../include/tensor.h"
#include <memory>
#include <cmath>
#include <vector>
#include <stdexcept>

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input)
{
    const std::vector<float>& in_data = input->data();
    std::size_t numel = input->numel();

    if (input->shape().empty() || (input->shape().size() == 1 && numel == 1))
    {
        float result = 1.0f;
        
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{input};

            std::function<void(const std::vector<float> &)> gradfn =
                [input](const std::vector<float> &grad_output)
            {
               
                std::vector<float> grad_input(input->numel(), 0.0f);
                input->add_to_grad(grad_input);
            };
           
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }

    if (input->shape().size() == 1)
    {
        float max_val = in_data[0];
        for (std::size_t i = 1; i < numel; i++)
        {
            if (in_data[i] > max_val) max_val = in_data[i];
        }

        std::vector<float> s(numel);
        float sum_exp = 0.0f;
        for (std::size_t i = 0; i < numel; i++)
        {
            float val = std::exp(in_data[i] - max_val);
            s[i] = val;
            sum_exp += val;
        }

        for (std::size_t i = 0; i < numel; i++)
        {
            s[i] /= sum_exp;
        }

        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{input};
            
            std::function<void(const std::vector<float> &)> gradfn =
                [input, s, numel](const std::vector<float> &grad_output)
            {
                std::vector<float> grad_input(numel, 0.0f);
                
                for (std::size_t j = 0; j < numel; j++)
                {
                    float grad_j = 0.0f;
                    float s_j = s[j];
                    
                    for (std::size_t i = 0; i < numel; i++)
                    {
                        if (i == j)
                            grad_j += grad_output[i] * s[i] * (1.0f - s_j); 
                        else
                            grad_j += grad_output[i] * (-s[i] * s_j);
                    }
                    grad_input[j] = grad_j;
                }
                
                input->add_to_grad(grad_input);
            };
            return std::make_shared<Tensor>(s, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(s);
    }
    throw std::runtime_error("Softmax is currently only allowed for scalars or 1D vectors.");
}
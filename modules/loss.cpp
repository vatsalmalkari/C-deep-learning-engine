#include "../include/loss.h"
#include "../include/module.h"
#include "../include/softmax.h"
#include "../include/tensor.h"
#include <algorithm>
#include <cmath>
#include <functional>

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input)
{
    throw std::runtime_error("Loss expects an inputs and target.");
}

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    throw std::runtime_error("Forward not implemented.");
}

std::shared_ptr<Tensor> Loss::operator()(std::shared_ptr<Tensor> input, std::size_t target)
{
    return forward(input, target);
}

std::shared_ptr<Tensor> NLLLoss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if (input->shape().size() != 1)
    {
        throw std::runtime_error("NLLLoss expects a 1d input tensor.");
    }
    if (target >= input->numel())
    {
        throw std::runtime_error("NLLLoss target out of bounds");
    }
    // prevent log(0)
    float prob = std::max((*input)(target), 1e-12f);
    float loss = -std::log(prob);

    if (input->requires_grad())
{
    std::vector<std::shared_ptr<Tensor>> parents{input};

    std::function<void(const std::vector<float>&)> gradfn = [input, target](const std::vector<float>& grad_output)
    {
        std::vector<float> grad_input(input->numel(), 0.0f);

        float eps = 1e-9f;
        grad_input[target] =
            grad_output[0] * (-1.0f / std::max((*input)(target), eps));

        input->add_to_grad(grad_input);
    };

    return std::make_shared<Tensor>(loss, true, gradfn, parents);
}
return std::make_shared<Tensor>(loss);


    }

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if (input->shape().size() != 1)
    {
        throw std::runtime_error("CrossEntropyLoss expects a 1d input tensor.");
    }
    if (target >= input->numel())
    {
        throw std::runtime_error("CrossEntropyLoss target out of bounds.");
    }

    std::size_t n = input->numel();
    const float* in_ptr = input->data();

    // 1. Numerically stable Log-Sum-Exp trick
    float max_val = *std::max_element(in_ptr, in_ptr + n);

    float sum_exp = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum_exp += std::exp(in_ptr[i] - max_val);
    }
    float log_sum_exp = max_val + std::log(sum_exp);

    // Loss = -log(softmax(x)[target]) = log_sum_exp - x[target]
    float loss = log_sum_exp - in_ptr[target];

    // 2. Precompute softmax probabilities for the backward step
    std::vector<float> probs(n);
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(in_ptr[i] - max_val) / sum_exp;
    }

    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};

        // Fused backward pass: dL/dz_i = p_i - y_i
        auto gradfn = [input, target, probs](const std::vector<float>& grad_output)
        {
            std::vector<float> grad_input = probs;
            grad_input[target] -= 1.0f; // Subtract 1 from the true target class

            float upstream_grad = grad_output[0];
            for (std::size_t i = 0; i < grad_input.size(); ++i) {
                grad_input[i] *= upstream_grad;
            }

            input->add_to_grad(grad_input);
        };

        return std::make_shared<Tensor>(loss, true, gradfn, parents);
    }

    return std::make_shared<Tensor>(loss);
}



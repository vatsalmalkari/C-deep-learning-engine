// pooling.cpp
#include "../include/pooling.h"
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cstring>

Pooling::Pooling(std::size_t kernel_size, std::size_t stride, Arena* arena)
    : _kernel_size(kernel_size), _stride(stride), _arena(arena)
{}

std::shared_ptr<Tensor> Pooling::forward(std::shared_ptr<Tensor> input)
{
    // Input Shape
    std::vector<std::size_t> in_shape = input->shape();
    if (in_shape.size() != 3)
        throw std::runtime_error("Pooling expects 3D input [Channels, Height, Width]");

    std::size_t C = in_shape[0];
    std::size_t H = in_shape[1];
    std::size_t W = in_shape[2];

    // Output Dimensions
    std::size_t H_out = (H - _kernel_size) / _stride + 1;
    std::size_t W_out = (W - _kernel_size) / _stride + 1;
    std::size_t out_numel = C * H_out * W_out;

    // Get raw data pointers
    const float* in_data = input->data();
    
    // Strides for flat indexing
    std::size_t in_stride_c = H * W;
    std::size_t in_stride_h = W;
    std::size_t out_stride_c = H_out * W_out;
    std::size_t out_stride_h = W_out;

    bool req_grad = input->requires_grad();

    // ALLOCATE OUTPUT FROM ARENA OR MALLOC
    std::shared_ptr<Tensor> output;
    std::vector<std::size_t> out_shape = {C, H_out, W_out};
    
    if (_arena) {
        Tensor* out_tensor = (Tensor*)arena_alloc(sizeof(Tensor), _arena);
        // Construct Tensor in-place using the same constructor used for non-arena allocation
        new (out_tensor) Tensor(std::vector<float>(out_numel, 0.0f), out_shape, req_grad);
        // FIX 2: Manually call destructor when ref count hits 0
        output = std::shared_ptr<Tensor>(out_tensor, [](Tensor* ptr) {
            if (ptr) ptr->~Tensor();
        });
    } else {
        // FIX 1: Pass req_grad to malloc constructor
        output = std::make_shared<Tensor>(std::vector<float>(out_numel, 0.0f), out_shape, req_grad);
    }
    
    float* out_data = output->data();

    // MAX POOLING FORWARD
    for (std::size_t c = 0; c < C; c++)
    {
        for (std::size_t h = 0; h < H_out; h++)
        {
            for (std::size_t w = 0; w < W_out; w++)
            {
                float max_val = -1e9f;
                
                std::size_t start_h = h * _stride;
                std::size_t start_w = w * _stride;

                for (std::size_t kh = 0; kh < _kernel_size; kh++)
                {
                    for (std::size_t kw = 0; kw < _kernel_size; kw++)
                    {
                        std::size_t cur_h = start_h + kh;
                        std::size_t cur_w = start_w + kw;
                        
                        std::size_t in_idx = c * in_stride_c + cur_h * in_stride_h + cur_w;
                        
                        if (in_data[in_idx] > max_val) {
                            max_val = in_data[in_idx];
                        }
                    }
                }
                
                std::size_t out_idx = c * out_stride_c + h * out_stride_h + w;
                out_data[out_idx] = max_val;
            }
        }
    }

    // ==========================================
    // BACKWARD PASS (Gradient Function)
    // ==========================================
    if (req_grad)
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};
        
        std::function<void(const std::vector<float>&)> gradfn = 
            [input, 
             C, H, H_out, W_out, 
             ks = _kernel_size, stride = _stride,
             in_stride_c, in_stride_h, out_stride_c, out_stride_h]
            (const std::vector<float>& grad_output)
        {
            std::vector<float> grad_input(input->size(), 0.0f);
            const float* in_vals = input->data();

            for (std::size_t c = 0; c < C; c++)
            {
                for (std::size_t h = 0; h < H_out; h++)
                {
                    for (std::size_t w = 0; w < W_out; w++)
                    {
                        float max_val = -1e9f;
                        std::size_t max_idx = 0;

                        std::size_t start_h = h * stride;
                        std::size_t start_w = w * stride;

                        for (std::size_t kh = 0; kh < ks; kh++)
                        {
                            for (std::size_t kw = 0; kw < ks; kw++)
                            {
                                std::size_t cur_h = start_h + kh;
                                std::size_t cur_w = start_w + kw;
                                std::size_t idx = c * in_stride_c + cur_h * in_stride_h + cur_w;
                                
                                if (in_vals[idx] > max_val) {
                                    max_val = in_vals[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        
                        std::size_t out_idx = c * out_stride_c + h * out_stride_h + w;
                        grad_input[max_idx] += grad_output[out_idx];
                    }
                }
            }
            input->add_to_grad(grad_input);
        };

        output->set_grad_fn(gradfn);
        output->set_parents(parents);
    }

    return output;
}

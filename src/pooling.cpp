#include "../include/pooling.h"
#include <limits>
#include <stdexcept>
#include <cmath>
#include <vector>

Pooling::Pooling(std::size_t kernel_size, std::size_t stride)
    : _kernel_size(kernel_size), _stride(stride)
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

    // Initialize Output as Flat Vector
    std::size_t out_numel = C * H_out * W_out;
    std::vector<float> out_data(out_numel, 0.0f);
    
    const std::vector<float>& in_data = input->data();
    
    // Strides for flat indexing
    std::size_t in_stride_c = H * W;
    std::size_t in_stride_h = W;
    std::size_t out_stride_c = H_out * W_out;
    std::size_t out_stride_h = W_out;

    // Max Pooling
    for (std::size_t c = 0; c < C; c++)
    {
        for (std::size_t h = 0; h < H_out; h++)
        {
            for (std::size_t w = 0; w < W_out; w++)
            {
                float max_val = -std::numeric_limits<float>::infinity();
                
                size_t start_h = h * _stride;
                size_t start_w = w * _stride;

                for (std::size_t kh = 0; kh < _kernel_size; kh++)
                {
                    for (std::size_t kw = 0; kw < _kernel_size; kw++)
                    {
                        size_t cur_h = start_h + kh;
                        size_t cur_w = start_w + kw;
                        
                        // Flat index calculation
                        size_t in_idx = c * in_stride_c + cur_h * in_stride_h + cur_w;
                        
                        if (in_data[in_idx] > max_val) {
                            max_val = in_data[in_idx];
                        }
                    }
                }
                
                size_t out_idx = c * out_stride_c + h * out_stride_h + w;
                out_data[out_idx] = max_val;
            }
        }
    }

    std::vector<std::size_t> out_shape = {C, H_out, W_out};

    //  Backward Pass
    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{input};
        
        std::function<void(const std::vector<float>&)> gradfn = 
            [input, 
             C, H, H_out, W_out, 
             ks=_kernel_size, stride=_stride,
             in_stride_c, in_stride_h, out_stride_c, out_stride_h]
            (const std::vector<float>& grad_output)
        {
            std::vector<float> grad_input(input->numel(), 0.0f);
            const std::vector<float>& in_vals = input->data();

            for (std::size_t c = 0; c < C; c++)
            {
                for (std::size_t h = 0; h < H_out; h++)
                {
                    for (std::size_t w = 0; w < W_out; w++)
                    {
                        // max index to route gradient
                        float max_val = -std::numeric_limits<float>::infinity();
                        size_t max_idx = 0;

                        size_t start_h = h * stride;
                        size_t start_w = w * stride;

                        for (std::size_t kh = 0; kh < ks; kh++)
                        {
                            for (std::size_t kw = 0; kw < ks; kw++)
                            {
                                size_t cur_h = start_h + kh;
                                size_t cur_w = start_w + kw;
                                size_t idx = c * in_stride_c + cur_h * in_stride_h + cur_w;
                                
                                if (in_vals[idx] > max_val) {
                                    max_val = in_vals[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        
                        size_t out_idx = c * out_stride_c + h * out_stride_h + w;
                        grad_input[max_idx] += grad_output[out_idx];
                    }
                }
            }
            input->add_to_grad(grad_input);
        };

        return std::make_shared<Tensor>(out_data, out_shape, true, gradfn, parents);
    }

    return std::make_shared<Tensor>(out_data, out_shape);
}
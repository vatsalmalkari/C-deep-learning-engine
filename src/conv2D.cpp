#include "../include/conv2d.h"
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

Conv2D::Conv2D(std::size_t in_channels, std::size_t out_channels, std::size_t kernel_size, std::size_t stride, std::size_t padding, std::size_t seed)
    : _in_channels(in_channels), _out_channels(out_channels),
      _kernel_size(kernel_size), _stride(stride), _padding(padding),
      _seed(seed)
{
    std::size_t weight_numel = out_channels * in_channels * kernel_size * kernel_size;
    std::vector<float> w(weight_numel);

    for (std::size_t i = 0; i < weight_numel; i++) {
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    std::vector<float> b(out_channels, 0.1f);

    _weight = std::make_shared<Tensor>(w, true);
    _bias = std::make_shared<Tensor>(b, true);

    register_parameter("weight", _weight);
    register_parameter("bias", _bias);
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input)
{
    bool should_create_graph = input->requires_grad() || _weight->requires_grad() || _bias->requires_grad();

    const auto& in_shape = input->shape(); 
    if (in_shape.size() != 3)
        throw std::runtime_error("Conv2D expects 3D input [C,H,W]");

    std::size_t C_in = in_shape[0];
    std::size_t H_in = in_shape[1];
    std::size_t W_in = in_shape[2];

    if (C_in != _in_channels)
         throw std::runtime_error("Input channels do not match Conv2D in_channels");

    std::size_t H_out = (H_in - _kernel_size + 2 * _padding) / _stride + 1;
    std::size_t W_out = (W_in - _kernel_size + 2 * _padding) / _stride + 1;
    std::size_t out_numel = _out_channels * H_out * W_out;

    std::vector<float> out(out_numel, 0.0f);

    const std::vector<float>& input_data = input->data();
    const std::vector<float>& weight_data = _weight->data();
    const std::vector<float>& bias_data = _bias->data();

    // Strides
    std::size_t input_stride_c = H_in * W_in;
    std::size_t input_stride_h = W_in;
    
    std::size_t weight_stride_co = _in_channels * _kernel_size * _kernel_size;
    std::size_t weight_stride_ci = _kernel_size * _kernel_size;
    std::size_t weight_stride_kh = _kernel_size;

    std::size_t out_stride_co = H_out * W_out;
    std::size_t out_stride_h = W_out;

    // --- Forward Pass ---
    for (std::size_t co = 0; co < _out_channels; co++) {
        float b_val = bias_data[co];
        for (std::size_t h = 0; h < H_out; h++) {
            for (std::size_t w = 0; w < W_out; w++) {
                float sum = b_val;
                for (std::size_t ci = 0; ci < C_in; ci++) {
                    for (std::size_t kh = 0; kh < _kernel_size; kh++) {
                        for (std::size_t kw = 0; kw < _kernel_size; kw++) {
                            int ih = h * _stride + kh - _padding;
                            int iw = w * _stride + kw - _padding;
                            
                            if (ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                std::size_t in_idx = ci * input_stride_c + ih * input_stride_h + iw;
                                std::size_t w_idx = co * weight_stride_co + ci * weight_stride_ci + kh * weight_stride_kh + kw;
                                sum += input_data[in_idx] * weight_data[w_idx];
                            }
                        }
                    }
                }
                out[co * out_stride_co + h * out_stride_h + w] = sum;
            }
        }
    }

    std::vector<std::size_t> out_shape = {_out_channels, H_out, W_out};

    if (should_create_graph)
    {
        std::vector<std::shared_ptr<Tensor>> parents{input, _weight, _bias};

        std::function<void(const std::vector<float>&)> gradfn =
            [input, weight=_weight, bias=_bias,
             C_in, H_in, W_in, 
             stride=_stride, padding=_padding, kernel_size=_kernel_size, C_out=_out_channels,
             H_out, W_out,
             input_stride_c, input_stride_h,
             weight_stride_co, weight_stride_ci, weight_stride_kh,
             out_stride_co, out_stride_h]
            (const std::vector<float>& grad_output_flat)
        {
            std::vector<float> grad_input(input->numel(), 0.0f);
            std::vector<float> grad_weight(weight->numel(), 0.0f);
            std::vector<float> grad_bias(bias->numel(), 0.0f);
            
            const std::vector<float>& w_data = weight->data();
            const std::vector<float>& in_data = input->data();

            // 1. Grad Bias
            for (std::size_t co = 0; co < C_out; co++) {
                float sum = 0.0f;
                for (std::size_t h = 0; h < H_out; h++) {
                    for (std::size_t w = 0; w < W_out; w++) {
                         sum += grad_output_flat[co * out_stride_co + h * out_stride_h + w];
                    }
                }
                grad_bias[co] = sum;
            }

            // 2. Grad Weights & Input
            for (std::size_t co = 0; co < C_out; co++) {
                for (std::size_t ci = 0; ci < C_in; ci++) {
                    for (std::size_t kh = 0; kh < kernel_size; kh++) {
                        for (std::size_t kw = 0; kw < kernel_size; kw++) {
                            float grad_w = 0.0f;
                            std::size_t w_idx = co * weight_stride_co + ci * weight_stride_ci + kh * weight_stride_kh + kw;
                            float w_val = w_data[w_idx];

                            for (std::size_t h = 0; h < H_out; h++) {
                                for (std::size_t w = 0; w < W_out; w++) {
                                    int ih = h * stride + kh - padding;
                                    int iw = w * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < (int)H_in && iw >= 0 && iw < (int)W_in) {
                                        std::size_t out_idx = co * out_stride_co + h * out_stride_h + w;
                                        std::size_t in_idx = ci * input_stride_c + ih * input_stride_h + iw;
                                        
                                        float g = grad_output_flat[out_idx];
                                        
                                        grad_w += in_data[in_idx] * g;
                                        grad_input[in_idx] += w_val * g;
                                    }
                                }
                            }
                            grad_weight[w_idx] = grad_w;
                        }
                    }
                }
            }

            // Safe update for input
            if (input->requires_grad()) {
                input->add_to_grad(grad_input);
            }
            weight->add_to_grad(grad_weight);
            bias->add_to_grad(grad_bias);
        };

        return std::make_shared<Tensor>(out, out_shape, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(out, out_shape);
}
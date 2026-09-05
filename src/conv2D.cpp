#include "../include/conv2d.h"
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

Conv2D::Conv2D(std::size_t in_channels, std::size_t out_channels, 
               std::size_t kernel_size, std::size_t stride, 
               std::size_t padding, ArenaAllocator* allocator, 
               std::size_t seed)
    : _in_channels(in_channels), _out_channels(out_channels),
      _kernel_size(kernel_size), _stride(stride), _padding(padding), 
      _allocator(allocator), _seed(seed) {
    
    std::size_t weight_numel = out_channels * in_channels * kernel_size * kernel_size;
    
    std::vector<float> w(weight_numel);
    float scale = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    for (std::size_t i = 0; i < weight_numel; i++) {
        w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    std::vector<float> b(out_channels, 0.0f);
    
    _weight = std::make_shared<Tensor>(
        w, 
        std::vector<std::size_t>{out_channels, in_channels, kernel_size, kernel_size}, 
        _allocator, 
        true
    );
    
    _bias = std::make_shared<Tensor>(
        b, 
        std::vector<std::size_t>{out_channels}, 
        _allocator, 
        true
    );
    
    register_parameter("weight", _weight);
    register_parameter("bias", _bias);
}

Conv2D::~Conv2D() = default;

void Conv2D::register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor) {
    _parameters.push_back({name, tensor});
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Conv2D::parameters() {
    return _parameters;
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> input) {
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

    const float* input_data = input->data();
    const float* weight_data = _weight->data();
    const float* bias_data = _bias->data();

    auto output = std::make_shared<Tensor>(
        std::vector<std::size_t>{_out_channels, H_out, W_out}, 
        _allocator,
        should_create_graph
    );
    
    float* out_data = output->data();

    std::size_t input_stride_c = H_in * W_in;
    std::size_t input_stride_h = W_in;
    
    std::size_t weight_stride_co = _in_channels * _kernel_size * _kernel_size;
    std::size_t weight_stride_ci = _kernel_size * _kernel_size;
    std::size_t weight_stride_kh = _kernel_size;

    std::size_t out_stride_co = H_out * W_out;
    std::size_t out_stride_h = W_out;

    // --- Forward Pass (Hoisted Row Calculation) ---
    for (std::size_t co = 0; co < _out_channels; co++) {
        float b_val = bias_data[co];
        for (std::size_t h = 0; h < H_out; h++) {
            for (std::size_t w = 0; w < W_out; w++) {
                float sum = b_val;
                for (std::size_t ci = 0; ci < C_in; ci++) {
                    std::size_t in_base_c = ci * input_stride_c;
                    std::size_t w_base_ci = co * weight_stride_co + ci * weight_stride_ci;

                    for (std::size_t kh = 0; kh < _kernel_size; kh++) {
                        int ih = static_cast<int>(h * _stride + kh - _padding);
                        if (ih < 0 || ih >= static_cast<int>(H_in)) continue;

                        std::size_t in_base_row = in_base_c + static_cast<std::size_t>(ih) * input_stride_h;
                        std::size_t w_base_row = w_base_ci + kh * weight_stride_kh;

                        for (std::size_t kw = 0; kw < _kernel_size; kw++) {
                            int iw = static_cast<int>(w * _stride + kw - _padding);
                            if (iw >= 0 && iw < static_cast<int>(W_in)) {
                                sum += input_data[in_base_row + static_cast<std::size_t>(iw)] * weight_data[w_base_row + kw];
                            }
                        }
                    }
                }
                out_data[co * out_stride_co + h * out_stride_h + w] = sum;
            }
        }
    }

    // --- Backward Pass ---
    if (should_create_graph) {
        std::vector<std::shared_ptr<Tensor>> parents{input, _weight, _bias};
        bool need_input_grad = input->requires_grad();

        std::function<void(const std::vector<float>&)> gradfn =
            [input, weight=_weight, bias=_bias, need_input_grad,
             C_in, H_in, W_in, 
             stride=_stride, padding=_padding, kernel_size=_kernel_size, C_out=_out_channels,
             H_out, W_out,
             input_stride_c, input_stride_h,
             weight_stride_co, weight_stride_ci, weight_stride_kh,
             out_stride_co, out_stride_h]
            (const std::vector<float>& grad_output_flat) {
                
            std::vector<float> grad_weight(weight->numel(), 0.0f);
            std::vector<float> grad_bias(bias->numel(), 0.0f);
            std::vector<float> grad_input;
            if (need_input_grad) {
                grad_input.assign(input->numel(), 0.0f);
            }
            
            const float* w_data = weight->data();
            const float* in_data = input->data();

            // 1. Bias Gradients
            for (std::size_t co = 0; co < C_out; co++) {
                float sum = 0.0f;
                std::size_t out_c_offset = co * out_stride_co;
                for (std::size_t h = 0; h < H_out; h++) {
                    std::size_t out_h_offset = out_c_offset + h * out_stride_h;
                    for (std::size_t w = 0; w < W_out; w++) {
                         sum += grad_output_flat[out_h_offset + w];
                    }
                }
                grad_bias[co] += sum;
            }

            // 2. Weight & Input Gradients (Hoisted Index Calculation)
            for (std::size_t co = 0; co < C_out; co++) {
                std::size_t out_c_offset = co * out_stride_co;
                std::size_t w_co_offset = co * weight_stride_co;

                for (std::size_t ci = 0; ci < C_in; ci++) {
                    std::size_t in_c_offset = ci * input_stride_c;
                    std::size_t w_ci_offset = w_co_offset + ci * weight_stride_ci;

                    for (std::size_t kh = 0; kh < kernel_size; kh++) {
                        std::size_t w_kh_offset = w_ci_offset + kh * weight_stride_kh;

                        for (std::size_t kw = 0; kw < kernel_size; kw++) {
                            float grad_w = 0.0f;
                            std::size_t w_idx = w_kh_offset + kw;
                            float w_val = w_data[w_idx];

                            for (std::size_t h = 0; h < H_out; h++) {
                                int ih = static_cast<int>(h * stride + kh - padding);
                                if (ih < 0 || ih >= static_cast<int>(H_in)) continue;

                                std::size_t out_h_offset = out_c_offset + h * out_stride_h;
                                std::size_t in_h_offset = in_c_offset + static_cast<std::size_t>(ih) * input_stride_h;

                                for (std::size_t w = 0; w < W_out; w++) {
                                    int iw = static_cast<int>(w * stride + kw - padding);
                                    if (iw >= 0 && iw < static_cast<int>(W_in)) {
                                        std::size_t in_idx = in_h_offset + static_cast<std::size_t>(iw);
                                        float g = grad_output_flat[out_h_offset + w];
                                        
                                        grad_w += in_data[in_idx] * g;
                                        if (need_input_grad) {
                                            grad_input[in_idx] += w_val * g;
                                        }
                                    }
                                }
                            }
                            grad_weight[w_idx] += grad_w;
                        }
                    }
                }
            }

            // Update gradients
            if (need_input_grad) {
                input->add_to_grad(grad_input);
            }
            weight->add_to_grad(grad_weight);
            bias->add_to_grad(grad_bias);
        };

        output->set_grad_fn(gradfn);
        output->set_parents(parents);
    }
    
    return output;
}

#pragma once
#include <vector>
#include <functional>



class Tensor : public std::enable_shared_from_this<Tensor>
{
    private:
    std::vector<float> _data;
    std:: vector<std::size_t> _shape;
    std:: vector<std::size_t> _stride;
    std::vector<float> _grad;
    bool _requires_grad = false;
    std::function<void(const std::vector<float> &)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    void _backward();
    bool _visited = false;
    void _reset_graph_visit();

    public:
    Tensor(float data, bool requires_grad = false, 
       std::function<void(const std::vector<float> &)> gradfn = {}, 
       std::vector<std::shared_ptr<Tensor>> parents = {});
       

    Tensor(std::vector<float> data, bool requires_grad = false, 
        std::function<void(const std::vector<float> &)> gradfn = {}, 
        std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<std::vector<float>> data, bool requires_grad = false, 
        std::function<void(const std::vector<float> &)> gradfn = {}, 
        std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<std::vector<std::vector<float>>> data, bool requires_grad = false,
        std::function<void(const std::vector<float> &)> gradfn = {},
        std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<std::vector<std::vector<std::vector<float>>>> data,
       bool requires_grad = false,
       std::function<void(const std::vector<float>&)> gradfn = {},
       std::vector<std::shared_ptr<Tensor>> parents = {});

    Tensor(std::vector<float> data, 
           std::vector<std::size_t> shape, 
           bool requires_grad = false, 
           std::function<void(const std::vector<float> &)> gradfn = {}, 
           std::vector<std::shared_ptr<Tensor>> parents = {});

    const float &item() const;
    float &item();
    const float &operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);
    const std::vector<std::size_t> &shape() const;
    const std::vector<std::size_t> &stride() const;
    const bool &requires_grad() const;
    const std::vector<float> &grad() const;
    void add_to_grad(const std::vector<float> &grad_update);
    void zero_grad();
    std::size_t numel() const;
    std::vector<float> &data();
    void backward();
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
    std::size_t argmax() const;
    
    // 3D access
float& operator()(size_t i, size_t j, size_t k) {
    size_t H = _shape[1];
    size_t W = _shape[2];
    return _data[i*H*W + j*W + k];
}

const float& operator()(size_t i, size_t j, size_t k) const {
    size_t H = _shape[1];
    size_t W = _shape[2];
    return _data[i*H*W + j*W + k];
}

// 4D access
float& operator()(size_t i, size_t j, size_t k, size_t l) {
    size_t C = _shape[1];
    size_t H = _shape[2];
    size_t W = _shape[3];
    return _data[i*C*H*W + j*H*W + k*W + l];
}

const float& operator()(size_t i, size_t j, size_t k, size_t l) const {
    size_t C = _shape[1];
    size_t H = _shape[2];
    size_t W = _shape[3];
    return _data[i*C*H*W + j*H*W + k*W + l];
}

    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};
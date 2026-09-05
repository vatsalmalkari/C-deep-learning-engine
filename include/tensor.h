#pragma once


#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "../include/arena_wrapper.h"

class Tensor : public std::enable_shared_from_this<Tensor>
{
private:
    // Contiguous tensor data
    float* _data = nullptr;
    std::size_t _size = 0;

    // Tensor metadata
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;

    // Autograd
    bool _requires_grad = false;
    std::vector<float> _grad;
    std::function<void(const std::vector<float>&)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _visited = false;

    // Memory allocator (pointer to external allocator)
    ArenaAllocator* _allocator = nullptr;

public:
    // ========== CONSTRUCTORS ==========

    // Scalar
    Tensor(float data,
           bool requires_grad = false,
           std::function<void(const std::vector<float>&)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    // 1D vector
    Tensor(const std::vector<float>& data,
           bool requires_grad = false,
           std::function<void(const std::vector<float>&)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    // 2D matrix
    Tensor(const std::vector<std::vector<float>>& data,
           bool requires_grad = false,
           std::function<void(const std::vector<float>&)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    // Flat data + explicit shape
    Tensor(const std::vector<float>& data,
           const std::vector<std::size_t>& shape,
           bool requires_grad = false,
           std::function<void(const std::vector<float>&)> gradfn = nullptr,
           std::vector<std::shared_ptr<Tensor>> parents = {});

    // Shape-only tensor with optional allocator
    Tensor(const std::vector<std::size_t>& shape,
           ArenaAllocator* allocator = nullptr,
           bool requires_grad = false);

    // Data + shape + optional allocator
    Tensor(const std::vector<float>& data,
           const std::vector<std::size_t>& shape,
           ArenaAllocator* allocator,
           bool requires_grad = false);

    // Deprecated compatibility constructor
    Tensor(const std::vector<std::size_t>& shape,
           bool requires_grad);

    ~Tensor();

    // ========== ACCESSORS ==========

    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& stride() const { return _stride; }

    std::size_t size() const { return _size; }
    std::size_t numel() const { return _size; }

    float* data() { return _data; }
    const float* data() const { return _data; }

    const std::vector<float>& grad() const { return _grad; }

    bool requires_grad() const { return _requires_grad; }

    // ========== ELEMENT ACCESS ==========

    float item() const;
    float& item();

    float& operator()(std::size_t i);
    const float& operator()(std::size_t i) const;

    float& operator()(std::size_t i, std::size_t j);
    const float& operator()(std::size_t i, std::size_t j) const;

    std::size_t argmax() const;

    // ========== AUTOGRAD ==========

    void zero_grad();

    void add_to_grad(const std::vector<float>& grad_update);

    void backward();

    void set_grad_fn(
        std::function<void(const std::vector<float>&)> gradfn)
    {
        _gradfn = gradfn;
    }

    void set_parents(
        const std::vector<std::shared_ptr<Tensor>>& parents)
    {
        _parents = parents;
    }

    std::shared_ptr<Tensor> operator+(
        std::shared_ptr<Tensor> other);

    std::shared_ptr<Tensor> operator*(
        std::shared_ptr<Tensor> other);

private:
    // Memory management
    void _allocate_data(std::size_t size);
    void _copy_data(const float* src, std::size_t size);

    // Shape/stride management
    void _initialize_strides();

    // Autograd graph traversal
    void _backward();
    void _reset_graph_visit();
};

std::ostream& operator<<(std::ostream& os, const Tensor& obj);
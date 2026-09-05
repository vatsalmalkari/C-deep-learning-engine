#include "../include/tensor.h"
#include "../include/allocator.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <ostream>
#include <cstring>

Arena tensor_arena;

// MEMORY MANAGEMENT

void Tensor::_allocate_data(std::size_t size) {
    if (_data != nullptr) {
        if (_allocator) {
            _allocator->free(_data);
        } else {
            free(_data);
        }
    }
    
    _size = size;
    
    if (_allocator) {
        _data = (float*)_allocator->allocate(size * sizeof(float));
    } else {
        _data = (float*)malloc(size * sizeof(float));
    }
    
    if (!_data && size > 0) {
        throw std::runtime_error("Memory allocation failed");
    }
    
    memset(_data, 0, size * sizeof(float));
}

void Tensor::_copy_data(const float* src, std::size_t size) {
    _allocate_data(size);
    memcpy(_data, src, size * sizeof(float));
}

void Tensor::_initialize_strides() {
    if (!_shape.empty()) {
        _stride.resize(_shape.size());
        std::size_t stride = 1;
        for (int i = _shape.size() - 1; i >= 0; --i) {
            _stride[i] = stride;
            stride *= _shape[i];
        }
    } else {
        _stride = {1};
    }
}

// CONSTRUCTORS

// Scalar Constructor (0D)
Tensor::Tensor(float data, bool requires_grad, 
               std::function<void(const std::vector<float> &)> gradfn, 
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{}, _stride{}, _requires_grad(requires_grad), _gradfn(gradfn),
      _parents(parents), _allocator(nullptr)
{
    _allocate_data(1);
    _data[0] = data;
    if (_requires_grad) {
        zero_grad();
    }
}

// 1D Vector Constructor
Tensor::Tensor(const std::vector<float>& data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{data.size()}, _requires_grad(requires_grad), _gradfn(gradfn),
      _parents(parents), _allocator(nullptr)
{
    _initialize_strides();
    _allocate_data(data.size());
    memcpy(_data, data.data(), data.size() * sizeof(float));
    if (_requires_grad) {
        zero_grad();
    }
}

// 2D Matrix Constructor
Tensor::Tensor(const std::vector<std::vector<float>>& data, bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape{data.size(), data[0].size()}, _requires_grad(requires_grad), 
      _gradfn(gradfn), _parents(parents), _allocator(nullptr)
{
    // Check if dimensions match
    std::size_t n_expected_columns = data[0].size();
    for (std::size_t i = 0; i < data.size(); i++) {
        if (data[i].size() != n_expected_columns) {
            throw std::invalid_argument("Dimensions are inconsistent.");
        }
    }
    
    _initialize_strides();
    
    // Store in row major format like PyTorch and NumPy
    std::size_t total_size = data.size() * n_expected_columns;
    _allocate_data(total_size);
    
    std::size_t idx = 0;
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[i].size(); j++) {
            _data[idx++] = data[i][j];
        }
    }
    
    if (_requires_grad) {
        zero_grad();
    }
}

// Constructor for Flat Data + Explicit Shape
Tensor::Tensor(const std::vector<float>& data, const std::vector<std::size_t>& shape,
               bool requires_grad,
               std::function<void(const std::vector<float> &)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents)
    : _shape(shape), _requires_grad(requires_grad), _gradfn(gradfn), 
      _parents(parents), _allocator(nullptr)
{
    // Calculate strides based on shape
    _initialize_strides();
    
    // Validate size
    std::size_t expected_size = 1;
    for (std::size_t dim : _shape) {
        expected_size *= dim;
    }
    
    if (expected_size != data.size()) {
        throw std::invalid_argument("Tensor shape does not match data size");
    }
    
    _copy_data(data.data(), data.size());
    
    if (_requires_grad) {
        zero_grad();
    }
}

// Shape-only tensor with optional allocator
Tensor::Tensor(const std::vector<std::size_t>& shape, ArenaAllocator* allocator, bool requires_grad)
    : _shape(shape), _requires_grad(requires_grad), _allocator(allocator)
{
    std::size_t size = 1;
    for (std::size_t dim : shape) {
        size *= dim;
    }
    
    _initialize_strides();
    _allocate_data(size);
    
    if (_requires_grad) {
        zero_grad();
    }
}

// Data + shape + optional allocator
Tensor::Tensor(const std::vector<float>& data, const std::vector<std::size_t>& shape,
               ArenaAllocator* allocator, bool requires_grad)
    : _shape(shape), _requires_grad(requires_grad), _allocator(allocator)
{
    // Calculate expected size
    std::size_t expected_size = 1;
    for (std::size_t dim : shape) {
        expected_size *= dim;
    }
    
    if (expected_size != data.size()) {
        throw std::invalid_argument("Tensor shape does not match data size");
    }
    
    _initialize_strides();
    _copy_data(data.data(), data.size());
    
    if (_requires_grad) {
        zero_grad();
    }
}

// Deprecated compatibility constructor
Tensor::Tensor(const std::vector<std::size_t>& shape, bool requires_grad)
    : Tensor(shape, nullptr, requires_grad) {}

// Destructor
Tensor::~Tensor() {
    if (_data != nullptr) {
        if (_allocator) {
            _allocator->free(_data);
        } else {
            free(_data);
        }
    }
}

// ========== ACCESSORS AND INDEXING ==========

float Tensor::item() const {
    if (_size == 1) {
        return _data[0];
    } else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

float& Tensor::item() {
    if (_size == 1) {
        return _data[0];
    } else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const float& Tensor::operator()(std::size_t i) const {
    if (_shape.size() == 0) {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1) {
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) + 
                                      " is out of bounds for array of size " + 
                                      std::to_string(_shape[0]));
        }
        return _data[i * _stride[0]];
    }
    throw std::invalid_argument("This is not a 1D tensor. Use appropriate indices.");
}

float& Tensor::operator()(std::size_t i) {
    if (_shape.size() == 0) {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1) {
        if (i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) + 
                                      " is out of bounds for array of size " + 
                                      std::to_string(_shape[0]));
        }
        return _data[i * _stride[0]];
    }
    throw std::invalid_argument("This is not a 1D tensor. Use appropriate indices.");
}

const float& Tensor::operator()(std::size_t i, std::size_t j) const {
    if (_shape.size() == 2) {
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) + 
                                      " is out of bounds for tensor with " + 
                                      std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) + 
                                      " is out of bounds for tensor with " + 
                                      std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

float& Tensor::operator()(std::size_t i, std::size_t j) {
    if (_shape.size() == 2) {
        if (i >= _shape[0]) {
            throw std::invalid_argument("Row index " + std::to_string(i) + 
                                      " is out of bounds for tensor with " + 
                                      std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1]) {
            throw std::invalid_argument("Column index " + std::to_string(j) + 
                                      " is out of bounds for tensor with " + 
                                      std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

std::size_t Tensor::argmax() const {
    if (_size == 0) {
        throw std::runtime_error("argmax() requires a non-empty tensor");
    }
    std::size_t max_idx = 0;
    float max_val = _data[0];
    for (std::size_t i = 1; i < _size; i++) {
        if (_data[i] > max_val) {
            max_val = _data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ========== MATH OPERATORS ==========

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    // scalar + scalar
    if (_shape.size() == 0 && other->shape().size() == 0) {
        float result = item() + other->item();
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // scalar + 1D
    if (_shape.size() == 0 && other->shape().size() == 1) {
        std::vector<float> result(other->shape()[0]);
        for (std::size_t i = 0; i < other->shape()[0]; i++) {
            result[i] = item() + (*other)(i);
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++) {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_self});
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // scalar + 2D
    if (_shape.size() == 0 && other->shape().size() == 2) {
        std::vector<float> result(_size = other->numel());
        for (std::size_t i = 0; i < other->numel(); i++) {
            result[i] = item() + other->data()[i];
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                float grad_self = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++) {
                    grad_self += grad_output[i];
                }
                self->add_to_grad({grad_self});
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, other->shape(), true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result, other->shape());
    }
    
    // 1D + scalar
    if (_shape.size() == 1 && other->shape().size() == 0) {
        std::vector<float> result(_size);
        for (std::size_t i = 0; i < _size; i++) {
            result[i] = _data[i] + other->item();
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                self->add_to_grad(grad_output);
                float grad_other = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++) {
                    grad_other += grad_output[i];
                }
                other->add_to_grad({grad_other});
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // 2D + scalar
    if (_shape.size() == 2 && other->shape().size() == 0) {
        std::vector<float> result(_size);
        for (std::size_t i = 0; i < _size; i++) {
            result[i] = _data[i] + other->item();
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                self->add_to_grad(grad_output);
                float grad_other = 0.0f;
                for (std::size_t i = 0; i < grad_output.size(); i++) {
                    grad_other += grad_output[i];
                }
                other->add_to_grad({grad_other});
            };
            return std::make_shared<Tensor>(result, _shape, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result, _shape);
    }
    
    // 1D + 1D
    if (_shape.size() == 1 && other->shape().size() == 1) {
        if (_shape[0] != other->shape()[0]) {
            throw std::invalid_argument("First dimensions are not equal.");
        }
        std::vector<float> result(_size);
        for (std::size_t i = 0; i < _size; i++) {
            result[i] = _data[i] + other->data()[i];
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // 2D + 2D
    if (_shape.size() == 2 && other->shape().size() == 2) {
        if (_shape[0] != other->shape()[0] || _shape[1] != other->shape()[1]) {
            throw std::invalid_argument("Tensor dimensions are not equal.");
        }
        std::vector<float> result(_size);
        for (std::size_t i = 0; i < _size; i++) {
            result[i] = _data[i] + other->data()[i];
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, _shape, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result, _shape);
    }
    
    throw std::invalid_argument("Addition not implemented for these shapes.");
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    if (_shape.size() == 0 || other->shape().size() == 0) {
        throw std::invalid_argument("Both arguments need to be at least 1D for matmul.");
    }
    if (_shape[_shape.size() - 1] != other->shape()[0]) {
        throw std::invalid_argument(
            "Last dimension of first tensor doesn't have same size as first dimension of second.");
    }
    
    // 1D x 1D -> scalar
    if (_shape.size() == 1 && other->shape().size() == 1) {
        float result = 0.0f;
        for (std::size_t i = 0; i < _size; i++) {
            result += _data[i] * other->data()[i];
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            const std::size_t size = _size;
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other, size](const std::vector<float> &grad_output) {
                std::vector<float> grad_self(size);
                std::vector<float> grad_other(other->numel());
                for (std::size_t i = 0; i < size; i++) {
                    grad_self[i] = other->data()[i] * grad_output[0];
                    grad_other[i] = self->data()[i] * grad_output[0];
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // 2D x 1D -> 1D
    if (_shape.size() == 2 && other->shape().size() == 1) {
        std::vector<float> result(_shape[0]);
        for (std::size_t i = 0; i < _shape[0]; i++) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < _shape[1]; j++) {
                sum += (*this)(i, j) * (*other)(j);
            }
            result[i] = sum;
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                std::vector<float> grad_self(self->numel());
                for (std::size_t i = 0; i < self->shape()[0]; i++) {
                    for (std::size_t j = 0; j < self->shape()[1]; j++) {
                        grad_self[i * self->shape()[1] + j] = 
                            (*other)(j) * grad_output[i];
                    }
                }
                std::vector<float> grad_other(other->numel());
                for (std::size_t j = 0; j < other->numel(); j++) {
                    float sum = 0.0f;
                    for (std::size_t i = 0; i < self->shape()[0]; i++) {
                        sum += (*self)(i, j) * grad_output[i];
                    }
                    grad_other[j] = sum;
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // 1D x 2D -> 1D
    if (_shape.size() == 1 && other->shape().size() == 2) {
        std::vector<float> result(other->shape()[1]);
        for (std::size_t j = 0; j < other->shape()[1]; j++) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < other->shape()[0]; i++) {
                sum += (*this)(i) * (*other)(i, j);
            }
            result[j] = sum;
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                std::vector<float> grad_self(self->numel());
                for (std::size_t i = 0; i < self->numel(); i++) {
                    float sum = 0.0f;
                    for (std::size_t j = 0; j < other->shape()[1]; j++) {
                        sum += (*other)(i, j) * grad_output[j];
                    }
                    grad_self[i] = sum;
                }
                std::vector<float> grad_other(other->numel());
                for (std::size_t i = 0; i < other->shape()[0]; i++) {
                    for (std::size_t j = 0; j < other->shape()[1]; j++) {
                        grad_other[i * other->shape()[1] + j] = 
                            (*self)(i) * grad_output[j];
                    }
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    
    // 2D x 2D
    if (_shape.size() == 2 && other->shape().size() == 2) {
        std::vector<float> result(_shape[0] * other->shape()[1]);
        for (std::size_t i = 0; i < _shape[0]; i++) {
            for (std::size_t j = 0; j < other->shape()[1]; j++) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < _shape[1]; k++) {
                    sum += (*this)(i, k) * (*other)(k, j);
                }
                result[i * other->shape()[1] + j] = sum;
            }
        }
        if (_requires_grad || other->requires_grad()) {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{self, other};
            std::function<void(const std::vector<float> &)> gradfn = 
                [self, other](const std::vector<float> &grad_output) {
                std::vector<float> grad_self(self->numel());
                for (std::size_t i = 0; i < self->shape()[0]; i++) {
                    for (std::size_t j = 0; j < self->shape()[1]; j++) {
                        float sum = 0.0f;
                        for (std::size_t k = 0; k < other->shape()[1]; k++) {
                            sum += (*other)(j, k) * 
                                   grad_output[i * other->shape()[1] + k];
                        }
                        grad_self[i * self->shape()[1] + j] = sum;
                    }
                }
                std::vector<float> grad_other(other->numel());
                for (std::size_t i = 0; i < other->shape()[0]; i++) {
                    for (std::size_t j = 0; j < other->shape()[1]; j++) {
                        float sum = 0.0f;
                        for (std::size_t k = 0; k < self->shape()[0]; k++) {
                            sum += (*self)(k, i) * 
                                   grad_output[k * other->shape()[1] + j];
                        }
                        grad_other[i * other->shape()[1] + j] = sum;
                    }
                }
                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, 
                                          std::vector<std::size_t>{_shape[0], other->shape()[1]},
                                          true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result, 
                                       std::vector<std::size_t>{_shape[0], other->shape()[1]});
    }
    
    throw std::invalid_argument("Matmul not implemented for these shapes.");
}

// AUTOGRAD

void Tensor::backward() {
    
    if (_shape.size() != 0) { 
        throw std::runtime_error("Grad can only be calculated for scalar outputs.");
    }
    
    _reset_graph_visit();
    _grad = std::vector<float>(1, 1.0f);
    _backward();
}

void Tensor::_backward() {
    if (!_requires_grad) {
        return;
    }
    if (_visited) {
        return;
    }
    _visited = true;
    if (_gradfn) {
        _gradfn(_grad);
    }
    for (std::size_t i = 0; i < _parents.size(); i++) {
        _parents[i]->_backward();
    }
}


void Tensor::add_to_grad(const std::vector<float> &grad_update) {
    if (!_requires_grad) {
        return;
    }
    if (_grad.size() != grad_update.size()) {
        throw std::runtime_error("Gradient shape mismatch during accumulation.");
    }
    for (std::size_t i = 0; i < _grad.size(); i++) {
        _grad[i] += grad_update[i];
    }
}

void Tensor::zero_grad() {
    _grad = std::vector<float>(_size, 0.0f);
}

void Tensor::_reset_graph_visit() {
    if (!_visited) {
        return;
    }
    _visited = false;
    for (std::size_t i = 0; i < _parents.size(); i++) {
        _parents[i]->_reset_graph_visit();
    }
}

//PRINTING 

std::ostream &operator<<(std::ostream &os, const Tensor &obj) {
    std::string string_repr = "[";
    if (obj.shape().size() == 0) {
        os << obj.item();
        return os;
    } else if (obj.shape().size() == 1) {
        for (std::size_t i = 0; i < obj.shape()[0]; i++) {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1) {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    } else if (obj.shape().size() == 2) {
        for (std::size_t i = 0; i < obj.shape()[0]; i++) {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++) {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1) {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1) {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}
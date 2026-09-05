#pragma once

#include "../include/allocator.h"
#include <stdexcept>
#include <cstring>

class ArenaAllocator {
private:
    Arena _arena;
    bool _initialized;

public:
    ArenaAllocator() : _initialized(false) {}
    
    ArenaAllocator(void* buffer, size_t size) : _initialized(false) {
        if (buffer == nullptr || size == 0) {
            throw std::invalid_argument("Arena buffer and size must be valid");
        }
        _arena = arena_init(buffer, size);
        _initialized = true;
    }
    
    ~ArenaAllocator() {
        if (_initialized) {
            arena_free_all(&_arena);
        }
    }
    
    // Delete copy operations
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    
    // Allow move operations
    ArenaAllocator(ArenaAllocator&& other) noexcept 
        : _arena(other._arena), _initialized(other._initialized) {
        other._initialized = false;
    }
    
    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept {
        if (this != &other) {
            if (_initialized) {
                arena_free_all(&_arena);
            }
            _arena = other._arena;
            _initialized = other._initialized;
            other._initialized = false;
        }
        return *this;
    }
    
    void* allocate(size_t size) {
        if (!_initialized) {
            throw std::runtime_error("Arena not initialized");
        }
        void* ptr = arena_alloc(size, &_arena);
        if (ptr == nullptr && size > 0) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    void free(void* ptr) {
        if (!_initialized) return;
        // Note: C arena doesn't track sizes, so pass 0
        arena_free(0, ptr, &_arena);
    }
    
    void free_all() {
        if (_initialized) {
            arena_free_all(&_arena);
        }
    }
    
    bool is_initialized() const { return _initialized; }
    
    size_t used() const { return _initialized ? _arena.offset : 0; }
    size_t total() const { return _initialized ? _arena.size : 0; }
    size_t remaining() const { 
        return _initialized ? (_arena.size - _arena.offset) : 0; 
    }
};
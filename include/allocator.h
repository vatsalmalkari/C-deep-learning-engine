#pragma once
#include <cstddef>
#include <cstdint>

// Alignment for cache-friendly memory (64 bytes for M3)
#define DEFAULT_ALIGNMENT 64

typedef struct {
    void* base;
    size_t size;
    size_t offset;
    size_t committed;
    
} Arena;

// Forward declarations
void* arena_alloc(size_t size, void* context);
void arena_free(size_t size, void* ptr, void* context);
void arena_free_all(void* context);
Arena arena_init(void* buffer, size_t size);


// Allocator interface
typedef struct {
    void* (*alloc)(size_t size, void* context);
    void (*free)(size_t size, void* ptr, void* context);
    void* context;
} Allocator;

#define arena_alloc_init(a) (Allocator){arena_alloc, arena_free, a}
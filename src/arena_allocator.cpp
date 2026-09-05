#include "../include/allocator.h"
#include <cstdint>

static bool is_power_of_two(size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

static uintptr_t align_forward(uintptr_t ptr, size_t alignment) {
    if (!is_power_of_two(alignment)){
        return 0;
    }
    uintptr_t a = alignment;
    uintptr_t modulo = ptr & (a - 1);
    if (modulo){ 
        ptr += a - modulo;
    }
    return ptr;
}

void* arena_alloc_aligned(Arena* a, size_t size, size_t alignment) {
    uintptr_t curr_ptr = (uintptr_t)a->base + (uintptr_t)a->offset;
    uintptr_t aligned_ptr = align_forward(curr_ptr, alignment);
    
    // Safety check for invalid alignment parameters
    if (aligned_ptr == 0) {
        return nullptr;
    }

    uintptr_t offset = aligned_ptr - (uintptr_t)a->base;
    
    if (offset + size > a->size) {
        return nullptr;  // Arena full
    }
    
    // Fix: Calculate total bytes consumed (including padding)
    size_t bytes_consumed = (offset + size) - a->offset;
    a->committed += bytes_consumed;
    
    void* ptr = (uint8_t*)a->base + offset;
    a->offset = offset + size;
    return ptr;
}

void* arena_alloc(size_t size, void* context) {
    if (!size){
        return nullptr;
    }
    return arena_alloc_aligned((Arena*)context, size, DEFAULT_ALIGNMENT);
}

void arena_free(size_t size, void* ptr, void* context) {
    (void)size; 
    (void)ptr; 
    (void)context;
}

void arena_free_all(void* context) {
    Arena* a = (Arena*)context;
    a->offset = 0;
    a->committed = 0;
}

Arena arena_init(void* buffer, size_t size) {
    Arena a;
    a.base = buffer;
    a.size = size;
    a.offset = 0;
    a.committed = 0;
    return a;
}

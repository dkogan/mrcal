#pragma once

/*
A simple min-priority-queue implementation. Uses STL internally

This is for the most part an internal implementation detail of
mrcal_traverse_sensor_links(), but could potentially be useful somewhere
on its own, so I'm exposing it here
 */


#include <stdint.h>
#include <stdbool.h>

typedef struct
{
    uint16_t idx_parent;
    bool     done : 1;
    uint64_t cost : 47;
} mrcal_heap_node_t;
// I can't find a single static assertion invocation that works in both C++ and
// C. The below is ugly, but works
#ifdef __cplusplus
static_assert( sizeof(mrcal_heap_node_t) == 8, "mrcal_heap_node_t has expected size");
#else
_Static_assert(sizeof(mrcal_heap_node_t) == 8, "mrcal_heap_node_t has expected size");
#endif

typedef struct
{
    uint16_t* buffer; // each of these indexes an external mrcal_heap_node_t[] array, which
                      // contains the cost
    int       size;
} mrcal_heap_t;

bool     mrcal_heap_empty (mrcal_heap_t* heap, mrcal_heap_node_t* nodes);
void     mrcal_heap_push  (mrcal_heap_t* heap, mrcal_heap_node_t* nodes, uint16_t x);
uint16_t mrcal_heap_pop   (mrcal_heap_t* heap, mrcal_heap_node_t* nodes);
void     mrcal_heap_resort(mrcal_heap_t* heap, mrcal_heap_node_t* nodes);

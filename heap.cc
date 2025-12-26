/*
  A simple min-priority-queue implementation. Uses STL internally
*/
#include <algorithm>
#include <functional>

extern "C"
{
#include "heap.h"
}

// empty namespace to prevent C++ from exporting any symbols here
namespace {
struct Compare_nodes_greater
{
    mrcal_heap_node_t* nodes;
    Compare_nodes_greater(mrcal_heap_node_t* _nodes) : nodes(_nodes) {}
    bool operator()(const uint16_t& idx_a, const uint16_t& idx_b) const
    {
        return nodes[idx_a].cost > nodes[idx_b].cost;
    }
};
}

extern "C"
bool     mrcal_heap_empty (mrcal_heap_t* heap, mrcal_heap_node_t* nodes)
{
    return heap->size==0;
}

extern "C"
void     mrcal_heap_push  (mrcal_heap_t* heap, mrcal_heap_node_t* nodes, uint16_t x)
{
    heap->buffer[heap->size++] = x;
    std::push_heap(&heap->buffer[0],
                   &heap->buffer[heap->size],
                   Compare_nodes_greater(nodes));
}

extern "C"
uint16_t mrcal_heap_pop   (mrcal_heap_t* heap, mrcal_heap_node_t* nodes)
{
    uint16_t x = heap->buffer[0];
    std::pop_heap(&heap->buffer[0],
                  &heap->buffer[heap->size],
                  Compare_nodes_greater(nodes));
    heap->size--;
    return x;
}

extern "C"
void mrcal_heap_resort(mrcal_heap_t* heap, mrcal_heap_node_t* nodes)
{
    std::make_heap(&heap->buffer[0],
                   &heap->buffer[heap->size],
                   Compare_nodes_greater(nodes));
}

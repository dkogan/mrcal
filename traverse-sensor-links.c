#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "mrcal.h"
#include "heap.h"

// assumes a != b
// test:
//   void main(void)
//   {
//       // 4x4 matrix
//       int A[] = {   1,2,3,
//                       4,5,
//                         6,
//                          };
//       for(int i=0; i<4; i++)
//       {
//           for(int j=0; j<4; j++)
//               if(i==j) printf("- ");
//               else     printf("%d ", A[pairwise_index(i,j,4)]);
//           printf("\n");
//       }
//   }
static int pairwise_index(const int a,
                          const int b,
                          const int N)
{
    // I have an (N,N) symmetric matrix with a 0 diagonal. I store the upper
    // triangle only, row-first: a 1D array of (N*(N-1)/2) values. This function
    // returns the linear index into this array
    //
    // If a > b: a + b*N - sum(1..b+1) = a + b*N - (b+1)*(b+2)/2
    if(a>b) return a + b*N - (b+1)*(b+2)/2;
    else    return b + a*N - (a+1)*(a+2)/2;
}

static
uint64_t cost_edge(uint16_t idx0, uint16_t idx1,
                   const uint16_t Nsensors,
                   const uint16_t* connectivity_matrix)
{
    // I want to MINIMIZE cost, so I MAXIMIZE the shared frames count and
    // MINIMIZE the hop count. Furthermore, I really want to minimize the number
    // of hops, so that's worth many shared frames (65536 of them)
    uint16_t num_shared_frames = connectivity_matrix[pairwise_index(idx0,idx1,Nsensors)];
    return 65536UL - (uint64_t)num_shared_frames;
}


static
void visit(const uint16_t idx,
           mrcal_heap_node_t* nodes,
           mrcal_heap_t* heap,
           const uint16_t Nsensors,
           const uint16_t* connectivity_matrix)
{
    mrcal_heap_node_t* node = &nodes[idx];
    node->done = true;

    for(int idx_neighbor=0; idx_neighbor<Nsensors; idx_neighbor++)
    {
        if(idx_neighbor == idx)
            continue;

        if(connectivity_matrix[pairwise_index(idx_neighbor,idx,Nsensors)] == 0)
            continue;

        mrcal_heap_node_t* node_neighbor = &nodes[idx_neighbor];
        if (node_neighbor->done)
            continue;

        uint64_t cost_to_neighbor_via_node =
            node->cost +
            cost_edge(idx_neighbor,idx,
                      Nsensors,
                      connectivity_matrix);

        if(node_neighbor->cost == 0)
        {
            // Haven't seen this node yet
            node_neighbor->cost       = cost_to_neighbor_via_node;
            node_neighbor->idx_parent = idx;
            mrcal_heap_push(heap, nodes, idx_neighbor);
        }
        else
        {
            // This node is already in the heap, ready to be processed.
            // If this new path to this node is better, use it
            if(cost_to_neighbor_via_node < node_neighbor->cost)
            {
                node_neighbor->cost       = cost_to_neighbor_via_node;
                node_neighbor->idx_parent = idx;
                mrcal_heap_resort(heap, nodes);
            }
        }
    }
}

bool mrcal_traverse_sensor_links( const uint16_t Nsensors,

                                        // (N,N) symmetric matrix with a 0 diagonal.
                                        // I store the upper triangle only,
                                        // row-first: a 1D array of (N*(N-1)/2)
                                        // values. use pairwise_index() to index
                                        const uint16_t* connectivity_matrix,
                                        const mrcal_callback_sensor_link_t cb,
                                        void* cookie)
{
    /*
    Traverses a connectivity graph of sensors

    Starts from the root sensor, and visits each one in order of total distance
    from the root. Useful to evaluate the whole set of sensors using pairwise
    metrics, building the network up from the best-connected, to the
    worst-connected. Any sensor not connected to the root at all will NOT be
    visited. The caller should check for any unvisited sensors.

    We have Nsensors sensors. Each one is identified by an integer in
    [0,Nsensors). The root is defined to be sensor 0.

    Three callbacks must be passed in:

    - callback__sensor_link(i, i_parent)
      Called when the best path to node i is found. This path runs through
      i_parent as the previous sensor

    */
    mrcal_heap_node_t nodes[Nsensors];
    memset(nodes,0,sizeof(nodes[0])*Nsensors);

    uint16_t heap_buffer[Nsensors];

    mrcal_heap_t heap = {.buffer = heap_buffer};

    visit(0,
          nodes,
          &heap,
          Nsensors, connectivity_matrix);

    while(!mrcal_heap_empty(&heap, nodes))
    {
        uint16_t idx_top = mrcal_heap_pop(&heap, nodes);

        if(!cb(idx_top, nodes[idx_top].idx_parent, cookie))
            return false;

        visit(idx_top,
              nodes,
              &heap,
              Nsensors, connectivity_matrix);
    }

    return true;
}

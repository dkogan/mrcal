// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "test-harness.h"
#include "../heap.h"

int main(int argc, char* argv[] )
{
    const int N = 10;

    uint16_t          heap_buffer[N];
    mrcal_heap_node_t nodes      [N];
    for(int i=0; i<N; i++)
    {
        nodes[i] = (mrcal_heap_node_t){.cost = i};
    }

    mrcal_heap_t heap = {.buffer = heap_buffer};

    mrcal_heap_push(&heap, nodes, 3);
    mrcal_heap_push(&heap, nodes, 2);
    mrcal_heap_push(&heap, nodes, 4);
    mrcal_heap_push(&heap, nodes, 1);
    mrcal_heap_push(&heap, nodes, 5);
    mrcal_heap_push(&heap, nodes, 9);


    uint16_t x;

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,1);

    nodes[2].cost += 10;
    mrcal_heap_resort(&heap, nodes);

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,3);

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,4);

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,5);

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,9);

    confirm(!mrcal_heap_empty(&heap, nodes));

    x = mrcal_heap_pop(&heap, nodes);
    confirm_eq_int(x,2);

    confirm(mrcal_heap_empty(&heap, nodes));

    TEST_FOOTER();
}

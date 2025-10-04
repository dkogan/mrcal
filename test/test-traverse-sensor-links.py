#!/usr/bin/env python3

r'''Checks the C and Python implementations of traverse_sensor_links()
'''

import sys
import numpy as np
import numpysane as nps
import os

testdir = os.path.dirname(os.path.realpath(__file__))

# I import the LOCAL mrcal since that's what I'm testing
sys.path[:0] = f"{testdir}/..",
import mrcal
import testutils

from mrcal.calibration import _traverse_sensor_links_python


connectivity_matrix = np.array((( 0,1,0,3,0),
                                ( 1,0,2,1,0),
                                ( 0,2,0,0,1),
                                ( 3,1,0,0,0),
                                ( 0,0,1,0,0),),
                               dtype=np.uint16)
Nsensors = connectivity_matrix.shape[0]




node_sequence        = []
node_sequence_python = []

mrcal.traverse_sensor_links(connectivity_matrix = connectivity_matrix,
                                  callback_sensor_link = lambda idx_to,idx_from: node_sequence.append((idx_to,idx_from),))


def cost_edge(camera_idx, from_idx):
    num_shared_frames = connectivity_matrix[camera_idx, from_idx]
    return 65536 - int(num_shared_frames) # int() to case up from uint16, and avoid an overflow
def neighbors(camera_idx):
    for neighbor_idx in range(Nsensors):
        if neighbor_idx == camera_idx                  or \
           connectivity_matrix[neighbor_idx,camera_idx] == 0:
            continue
        yield neighbor_idx


_traverse_sensor_links_python(Nsensors = Nsensors,
                                    callback__neighbors = neighbors,
                                    callback__cost_edge = cost_edge,
                                    callback__sensor_link = lambda idx_to,idx_from: node_sequence_python.append((idx_to,idx_from),))

testutils.confirm_equal( node_sequence,
                         node_sequence_python,
                         worstcase = True,
                         msg=f'The sequence of visited nodes matches in C and Python')

testutils.finish()

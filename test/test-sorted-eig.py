#!/usr/bin/env python3

r'''Tests mrcal.sorted_eig

This has complex logic to support broadcasting, and I validate it here
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


# Reference data generated like this:
#   l_ref = np.random.random((4,5,3))
#   v_ref = np.random.random((4,5,3,3))
#   v_ref /= nps.dummy(nps.mag(v_ref), -1)
l_ref = np.array(
      [[[0.67155361, 0.3739189 , 0.95940538],
        [0.14668176, 0.07336839, 0.86729039],
        [0.75139476, 0.44131877, 0.65641997],
        [0.08619787, 0.00514196, 0.46842911],
        [0.10580845, 0.32199031, 0.3735915 ]],
       [[0.05508103, 0.48401787, 0.46611417],
        [0.9483949 , 0.87735674, 0.45872956],
        [0.17057072, 0.47791177, 0.19352408],
        [0.13524878, 0.48731269, 0.42794007],
        [0.26253366, 0.55161897, 0.95356585]],
       [[0.70013654, 0.15301631, 0.66004348],
        [0.77747137, 0.48248562, 0.59674648],
        [0.67373556, 0.42192842, 0.50667046],
        [0.55300486, 0.99944748, 0.94900917],
        [0.23368724, 0.88016081, 0.95223845]],
       [[0.00710551, 0.59964954, 0.07766728],
        [0.3990046 , 0.11612546, 0.9119466 ],
        [0.58064238, 0.2557102 , 0.37509017],
        [0.8777226 , 0.49047888, 0.93402964],
        [0.69596428, 0.53620116, 0.17331565]]])
v_ref = np.array(
      [[[[0.25492998, 0.96651093, 0.0294505 ],
         [0.60080772, 0.41212806, 0.68496755],
         [0.73501109, 0.48565382, 0.47317974]],
        [[0.70813727, 0.03220643, 0.70533988],
         [0.93102532, 0.36255803, 0.04175557],
         [0.20480975, 0.40688981, 0.89022112]],
        [[0.7224457 , 0.24393336, 0.64696888],
         [0.35658574, 0.13414791, 0.9245815 ],
         [0.13909208, 0.60359174, 0.78506714]],
        [[0.75648707, 0.27895008, 0.59153543],
         [0.72671999, 0.43434694, 0.53218492],
         [0.65150949, 0.75369119, 0.08651575]],
        [[0.97809611, 0.20788551, 0.0105651 ],
         [0.54238903, 0.47683243, 0.69169717],
         [0.49810472, 0.82565147, 0.2649365 ]]],
       [[[0.61093683, 0.14117182, 0.77899083],
         [0.6181547 , 0.77847647, 0.10889971],
         [0.08704351, 0.1252583 , 0.98829843]],
        [[0.60466605, 0.73939429, 0.29609974],
         [0.16013313, 0.70419195, 0.69171604],
         [0.99552064, 0.07759479, 0.05401577]],
        [[0.82056474, 0.21231952, 0.53065425],
         [0.76537934, 0.56808399, 0.30244843],
         [0.946437  , 0.31493105, 0.07124214]],
        [[0.74296313, 0.65833599, 0.12082848],
         [0.55311763, 0.48774553, 0.67540002],
         [0.04022766, 0.231213  , 0.97207113]],
        [[0.74732957, 0.55426305, 0.36645734],
         [0.46692915, 0.62976623, 0.62078311],
         [0.33625853, 0.81507223, 0.47179175]]],
       [[[0.72795853, 0.68050259, 0.08362175],
         [0.69587492, 0.06301186, 0.71539332],
         [0.61449548, 0.78748595, 0.04755187]],
        [[0.0804386 , 0.97052257, 0.22719061],
         [0.68761638, 0.29892877, 0.66168369],
         [0.11544557, 0.60308161, 0.78928125]],
        [[0.70195868, 0.03183947, 0.71150563],
         [0.64728709, 0.59533928, 0.4760153 ],
         [0.7341157 , 0.52967069, 0.4248801 ]],
        [[0.87429733, 0.33296173, 0.35318645],
         [0.96062012, 0.27237033, 0.05498541],
         [0.50479801, 0.86215336, 0.04324986]],
        [[0.83840394, 0.5022181 , 0.21179193],
         [0.16494462, 0.94329994, 0.28805987],
         [0.09539021, 0.97808363, 0.18507601]]],
       [[[0.98847591, 0.1197682 , 0.09257946],
         [0.88339796, 0.34700512, 0.31495315],
         [0.36753448, 0.75206847, 0.54709362]],
        [[0.47027684, 0.63504071, 0.61283194],
         [0.92883661, 0.15765952, 0.33527009],
         [0.83514385, 0.04180379, 0.5484407 ]],
        [[0.85047215, 0.29296156, 0.43688746],
         [0.48273108, 0.84169098, 0.24192353],
         [0.41951382, 0.89475556, 0.15303802]],
        [[0.2493765 , 0.88257578, 0.39858669],
         [0.06407165, 0.67978738, 0.73060519],
         [0.78739648, 0.07009669, 0.61244856]],
        [[0.33123086, 0.91225007, 0.24101023],
         [0.02113432, 0.10669425, 0.99406724],
         [0.84421151, 0.19562951, 0.4990351 ]]]])

# I want the unit vectors in cols, not rows
v_ref = nps.transpose(v_ref)

@nps.broadcast_define( ( ('N',), ),
                       ('N','N'))
def diag(d):
    return np.diag(d)

X = nps.matmult(v_ref, diag(l_ref), np.linalg.inv(v_ref))

####### First, the simple non-broadcasted case
l,v = mrcal.sorted_eig(X[1,2])

testutils.confirm_equal(l.shape,
                        (3,),
                        msg = "Single matrix: l.shape")
testutils.confirm_equal(v.shape,
                        (3,3,),
                        msg = "Single matrix: v.shape")
testutils.confirm(np.all(np.diff(l) > 0),
                  msg = "Single matrix: monotonic eigenvalues")
testutils.confirm_equal(l,
                        np.sort(l_ref[1,2]),
                        worstcase = True,
                        eps=1e-6,
                        msg = "Single matrix: eigenvalues")
isorted = np.argsort(l_ref[1,2])
for i in range(len(isorted)):
    # I check the abs(inner) to ignore sign differences
    testutils.confirm_equal(np.abs(nps.inner(v[:,i],v_ref[1,2,:,isorted[i]])),
                            1.0,
                            worstcase = True,
                            eps=1e-6,
                            msg = f"Single matrix: eigenvector[{i}]")


####### Full, broadcasted case
l,v = mrcal.sorted_eig(X)

all_shapes_passed = \
    all((x for x in \
         (testutils.confirm_equal(l.shape,
                                  X.shape[:-1],
                                  msg = "Broadcasted matrix: l.shape"),
          testutils.confirm_equal(v.shape,
                                  X.shape,
                                  msg = "Broadcasted matrix: v.shape"))))
# I only bother checking the values if the shapes are right
if all_shapes_passed:
    testutils.confirm(np.all(np.diff(l) > 0),
                      msg = "Broadcasted matrix: monotonic eigenvalues")
    testutils.confirm_equal(l,
                            np.sort(l_ref),
                            worstcase = True,
                            eps=1e-6,
                            msg = "Broadcasted matrix: eigenvalues")

    # the complex eigenvector selection logic is the thing being tested, so I
    # check the reconstituted full matrix instead
    testutils.confirm_equal(nps.matmult(v, diag(l), np.linalg.inv(v)),
                            X,
                            worstcase = True,
                            eps=1e-6,
                            msg = f"Broadcasted matrix: eigenvectors")

testutils.finish()

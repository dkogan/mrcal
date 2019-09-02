#!/usr/bin/python3

import sys
import numpy as np
import os
import re
from inspect import currentframe

Nchecks = 0
NchecksFailed = 0

# no line breaks. Useful for test reporting. Yes, this sets global state, but
# we're running a test harness. This is fine
np.set_printoptions(linewidth=1e10, suppress=True)


def test_location():
    r'''Reports string describing current location in the test'''


    filename_this = os.path.split( __file__ )[1]

    frame = currentframe().f_back.f_back

    while frame:
        if frame.f_back is None or \
           not frame.f_code.co_filename.endswith(filename_this):
            break
        frame = frame.f_back

    testfile = os.path.split(frame.f_code.co_filename)[1]
    try:
        return "{}:{}".format(testfile, frame.f_lineno)
    except:
        return ''


def print_red(x):
    """print the message in red"""
    sys.stdout.write("\x1b[31m" + test_location() + ": " + x + "\x1b[0m\n")


def print_green(x):
    """Print the message in green"""
    sys.stdout.write("\x1b[32m" + test_location() + ": " + x + "\x1b[0m\n")


def confirm_equal(x, xref, msg='', eps=1e-6):
    r'''If x is equal to xref, report test success.

    msg identifies this check. eps sets the RMS equality tolerance. The x,xref
    arguments can be given as many different types. This function tries to do
    the right thing.

    '''

    global Nchecks
    global NchecksFailed
    Nchecks = Nchecks + 1

    # strip all trailing whitespace in each line, in case these are strings
    if isinstance(x, str):
        x = re.sub('[ \t]+(\n|$)', '\\1', x)
    if isinstance(xref, str):
        xref = re.sub('[ \t]+(\n|$)', '\\1', xref)

    # convert data to numpy if possible
    try:
        xref = np.array(xref)
    except:
        pass
    try:
        x = np.array(x)
    except:
        pass

    try:  # flatten array if possible
        x = x.ravel()
        xref = xref.ravel()
    except:
        pass

    try:
        N = x.shape[0]
    except:
        N = 1
    try:
        Nref = xref.shape[0]
    except:
        Nref = 1

    if N != Nref:

        print_red(("FAILED{}: mismatched array sizes: N = {} but Nref = {}. Arrays: \n" +
                   "x = {}\n" +
                   "xref = {}").
                  format((': ' + msg) if msg else '',
                         N, Nref,
                         x, xref))
        NchecksFailed = NchecksFailed + 1
        return False

    if N != 0:
        try:  # I I can subtract, get the error that way
            diff = x - xref

            def norm2sq(x):
                """Return 2 norm"""
                return np.inner(x, x)
            rms = np.sqrt(norm2sq(diff) / N)
            if not np.all(np.isfinite(rms)):
                print_red("FAILED{}: Some comparison results are NaN or Inf. "
                          "rms error = {}. x = {}, xref = {}".format(
                              (': ' + msg) if msg else '', rms, x, xref))
                NchecksFailed = NchecksFailed + 1
                return False
            if rms > eps:
                print_red("FAILED{}: rms error = {}.\nx,xref,err =\n{}".format(
                    (': ' + msg) if msg else '', rms,
                    np.vstack((x, xref, diff)).transpose()))
                NchecksFailed = NchecksFailed + 1
                return False
        except:  # Can't subtract. Do == instead
            if not np.array_equal(x, xref):
                print_red("FAILED{}: x =\n'{}', xref =\n'{}'".format(
                    (': ' + msg) if msg else '', x, xref))
                NchecksFailed = NchecksFailed + 1
                return False
    print_green("OK{}".format((': ' + msg) if msg else ''))
    return True


def confirm(x, msg=''):
    r'''If x is true, report test success.

    msg identifies this check'''

    global Nchecks
    global NchecksFailed
    Nchecks = Nchecks + 1

    if not x:
        print_red("FAILED{}".format((': ' + msg) if msg else ''))
        NchecksFailed = NchecksFailed + 1
        return False
    print_green("OK{}".format((': ' + msg) if msg else ''))
    return True


def finish():
    r'''Finalize the executed tests.

    Prints the test summary. Exits successfully iff all the tests passed.

    '''
    if not Nchecks and not NchecksFailed:
        print_red("No tests defined")
        sys.exit(0)

    if NchecksFailed:
        print_red("Some tests failed: {} out of {}".format(NchecksFailed, Nchecks))
        sys.exit(1)

    print_green("All tests passed: {} total".format(Nchecks))
    sys.exit(0)

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:29:21 2020

@author: willr
"""

cimport phaseCorr
import numpy as np

def test():
    return phaseCorr.testfunc()

def runSearch_GPU(Reference, Moving, params, batch_size):
    shape = np.zeros(2, dtype=np.int32)
    shape[0] = Reference.shape[0]
    shape[1] = Reference.shape[1]
    assert(Moving.shape == Reference.shape)
    
    cdef unsigned char[:] c_ref = np.asarray(Reference.ravel(), dtype=np.uint8)
    cdef unsigned char[:] c_mov = np.asarray(Moving.ravel(), dtype=np.uint8)
    
    cdef int[:] c_shape = shape;
    print(np.asarray(c_shape))
    cdef double[:] c_params = params
    
    cdef double[:] soln = np.zeros(3, dtype=np.float);
    cdef int c_batch_size = <int> batch_size;
    
    with nogil:
        phaseCorr.performSearch(c_batch_size, <unsigned char*> &c_ref[0], <unsigned char*> &c_mov[0], <int*> &c_shape[0], <double*> &c_params[0], <double*> &soln[0])
    
    s = np.asarray(soln)
    return (s[0], s[1], s[2]);
    

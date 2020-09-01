# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:27:32 2020

@author: willr
"""

cdef extern from "phaseCorrelation.h" nogil:
    int testfunc();
    void performSearch(int batch_size, unsigned char* reference, unsigned char* moving, int* shape, double* params, double* soln);

#pragma once

#include<winapifamily.h>

extern "C" __declspec(dllexport) void performSearch(unsigned int batch_size, unsigned char* reference, unsigned char* moving, int* shape, double* params, double* soln);

extern "C" __declspec(dllexport) int testfunc();

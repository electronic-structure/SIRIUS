#include "sirius.h"

int main(int argn, char** argv)
{
    complex16 a;
    gemm<cpu>(0, 2, 10, 10, 10, complex16(1.0, 0.0), &a, 10, &a, 10, complex16(0.0, 0.0), &a, 10); 
}



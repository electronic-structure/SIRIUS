// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file plasma_interface.cpp
 *   
 *  \brief PLASMA library interface. 
 */

#include <plasma.h>
#include <mkl_service.h>
//#include <plasma_threadsetting.h>

extern "C" void plasma_init(int num_cores)
{
    PLASMA_Init(num_cores);
}

extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval)
{
    PLASMA_desc* descT;

    PLASMA_Alloc_Workspace_zheevd(matrix_size, matrix_size, &descT);

    int info = PLASMA_zheevd(PlasmaVec, PlasmaUpper, matrix_size, (PLASMA_Complex64_t*)a, lda, eval, descT, (PLASMA_Complex64_t*)z, ldz);
    if (info != 0)
    {
        std::printf("erorr calling PLASMA_zheevd\n");
        exit(0);
    }

    PLASMA_Dealloc_Handle_Tile(&descT);
}


extern "C" void plasma_set_num_threads(int num_threads)
{
    //plasma_setlapack_numthreads(num_threads);
    mkl_set_num_threads(num_threads);
}

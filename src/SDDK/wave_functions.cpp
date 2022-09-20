// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wave_functions.cpp
 *
 *  \brief Definitions.
 *
 */
#include "wave_functions.hpp"

namespace sddk {

#if defined(SIRIUS_GPU)
void add_square_sum_gpu(std::complex<double> const* wf__, int num_rows_loc__, int nwf__, int reduced__, int mpi_rank__, double* result__)
{
    add_square_sum_gpu_double(wf__, num_rows_loc__, nwf__, reduced__, mpi_rank__, result__);
}

void add_square_sum_gpu(std::complex<float> const* wf__, int num_rows_loc__, int nwf__, int reduced__, int mpi_rank__, float* result__)
{
    add_square_sum_gpu_float(wf__, num_rows_loc__, nwf__, reduced__, mpi_rank__, result__);
}

void scale_matrix_columns_gpu(int nrow__, int ncol__, std::complex<double>* mtrx__, double* a__)
{
    scale_matrix_columns_gpu_double(nrow__, ncol__, mtrx__, a__);
}

void scale_matrix_columns_gpu(int nrow__, int ncol__, std::complex<float>* mtrx__, float* a__)
{
    scale_matrix_columns_gpu_float(nrow__, ncol__, mtrx__, a__);
}
#endif


} // namespace sddk



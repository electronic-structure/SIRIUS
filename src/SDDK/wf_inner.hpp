// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wf_inner.hpp
 *
 *  \brief Contains implementation of inner product for wave-functions.
 */
#ifndef __WF_INNER_HPP__
#define __WF_INNER_HPP__
#include "wave_functions.hpp"
#include <spla/spla.hpp>

namespace sddk {

/// Inner product between wave-functions.
/** This function computes the inner product using a moving window scheme plus allreduce.
 *  The input wave-functions data must be previously allocated on the GPU.
 *  The result is always returned in the CPU pointer.
 *
 *  The following \f$ m \times n \f$ sub-matrix is computed:
 *  \f[
 *    S_{irow0+i,jcol0+j} = \langle \phi_{i0 + i} | \tilde \phi_{j0 + j} \rangle
 *  \f]
 *
 *  \param [in]  spla_ctx Spla context
 *  \param [in]  ispn     Index of spin (0, 1, or 2; 2 means the contribution from two spinor components).
 *  \param [in]  bra      "bra" wave-functions \f$ \phi \f$.
 *  \param [in]  i0       Index of the first "bra" wave-function.
 *  \param [in]  m        Number of "bra" wave-functions.
 *  \param [in]  ket      "ket" wave-functions \f$ \tilde \phi \f$.
 *  \param [in]  j0       Index of the first "ket" wave-function.
 *  \param [in]  n        Number of "ket" wave-functions.
 *  \param [out] result   Resulting inner product matrix \f$ S \f$.
 *  \param [in]  irow0    First row (in the global matrix) of the inner product sub-matrix.
 *  \param [in]  jcol0    First column (in the global matix) of the inner product sub-matrix.
 */
template <typename T>
void inner(::spla::Context& spla_ctx__, spin_range ispn__, Wave_functions& bra__, int i0__, int m__,
           Wave_functions& ket__, int j0__, int n__, dmatrix<T>& result__, int irow0__, int jcol0__);
}
#endif

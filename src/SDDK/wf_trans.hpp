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

/** \file wf_trans.hpp
 *
 *  \brief Wave-function linear transformation.
 */
#ifndef __WF_TRANS_HPP__
#define __WF_TRANS_HPP__
#include "wave_functions.hpp"
namespace sddk {


/// Linear transformation of the wave-functions.
/** The transformation matrix is expected in the CPU memory. The following operation is performed:
 *  \f[
 *     \psi^{out}_{j} = \alpha \sum_{i} \psi^{in}_{i} Z_{ij} + \beta \psi^{out}_{j}
 *  \f]
 */
template <typename T>
void transform(memory_t                     mem__,
               linalg_t                     la__,
               int                          ispn__,
               double                       alpha__,
               std::vector<Wave_functions*> wf_in__,
               int                          i0__,
               int                          m__,
               dmatrix<T>&                  mtrx__,
               int                          irow0__,
               int                          jcol0__,
               double                       beta__,
               std::vector<Wave_functions*> wf_out__,
               int                          j0__,
               int                          n__);

template <typename T>
inline void transform(memory_t                     mem__,
                      linalg_t                     la__,
                      int                          ispn__,
                      std::vector<Wave_functions*> wf_in__,
                      int                          i0__,
                      int                          m__,
                      dmatrix<T>&                  mtrx__,
                      int                          irow0__,
                      int                          jcol0__,
                      std::vector<Wave_functions*> wf_out__,
                      int                          j0__,
                      int                          n__)
{
    transform<T>(mem__, la__, ispn__, 1.0, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, wf_out__, j0__, n__);
}

template <typename T>
inline void transform(memory_t        mem__,
                      linalg_t        la__,
                      int             ispn__,
                      Wave_functions& wf_in__,
                      int             i0__,
                      int             m__,
                      dmatrix<T>&     mtrx__,
                      int             irow0__,
                      int             jcol0__,
                      Wave_functions& wf_out__,
                      int             j0__,
                      int             n__)
{
    transform<T>(mem__, la__, ispn__, 1.0, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, {&wf_out__}, j0__, n__);
}

}
#endif

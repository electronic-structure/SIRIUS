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
#include <spla/spla.hpp>
#include "type_definition.hpp"
namespace sddk {

/// Linear transformation of the wave-functions.
/** The transformation matrix is expected in the CPU memory. The following operation is performed:
 *  \f[
 *     \psi^{out}_{j} = \alpha \sum_{i} \psi^{in}_{i} Z_{ij} + \beta \psi^{out}_{j}
 *  \f]
 */
template <typename T>
void transform(::spla::Context& spla_ctx__,
               int                                         ispn__,
               real_type<T>                                alpha__,
               std::vector<Wave_functions<real_type<T>>*>  wf_in__,
               int                                         i0__,
               int                                         m__,
               dmatrix<T>&                                 mtrx__,
               int                                         irow0__,
               int                                         jcol0__,
               real_type<T>                                beta__,
               std::vector<Wave_functions<real_type<T>>*>  wf_out__,
               int                                         j0__,
               int                                         n__);

inline void
transform(::spla::Context& spla_ctx__, int ispn__, double alpha__, std::vector<Wave_functions<float>*>  wf_in__,
          int i0__, int m__, dmatrix<std::complex<double>>& mtrx__, int irow0__, int jcol0__,
          double beta__, std::vector<Wave_functions<float>*>  wf_out__, int j0__, int n__)
{
    spin_range spins(ispn__);

    for (int idx = 0; idx < static_cast<int>(wf_in__.size()); idx++) {
        for (auto s : spins) {
	    for (int j = 0; j < n__; j++) {
	        for (int k = 0; k < wf_in__[idx]->pw_coeffs(s).num_rows_loc(); k++) {
		    std::complex<double> z(0, 0);;
		    for (int i = 0; i < m__; i++) {
			z += static_cast<std::complex<double>>(wf_in__[idx]->pw_coeffs(s).prime(k, i + i0__)) * mtrx__(irow0__ + i, jcol0__ + j);
		    }
		    wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__) = alpha__ * z + static_cast<std::complex<double>>(wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__)) * beta__;
		}
	    }
	}
    }

}

inline void
transform(::spla::Context& spla_ctx__, int ispn__, double alpha__, std::vector<Wave_functions<float>*>  wf_in__,
          int i0__, int m__, dmatrix<double>& mtrx__, int irow0__, int jcol0__,
          double beta__, std::vector<Wave_functions<float>*>  wf_out__, int j0__, int n__)
{
    spin_range spins(ispn__);

    for (int idx = 0; idx < static_cast<int>(wf_in__.size()); idx++) {
        for (auto s : spins) {
	    for (int j = 0; j < n__; j++) {
	        for (int k = 0; k < wf_in__[idx]->pw_coeffs(s).num_rows_loc(); k++) {
		    std::complex<double> z(0, 0);;
		    for (int i = 0; i < m__; i++) {
			z += static_cast<std::complex<double>>(wf_in__[idx]->pw_coeffs(s).prime(k, i + i0__)) * mtrx__(irow0__ + i, jcol0__ + j);
		    }
		    wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__) = alpha__ * z + static_cast<std::complex<double>>(wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__)) * beta__;
		}
	    }
	}
    }

}


template <typename T>
inline void transform(::spla::Context& spla_ctx__,
                      int                                        ispn__,
                      std::vector<Wave_functions<real_type<T>>*> wf_in__,
                      int                                        i0__,
                      int                                        m__,
                      dmatrix<T>&                                mtrx__,
                      int                                        irow0__,
                      int                                        jcol0__,
                      std::vector<Wave_functions<real_type<T>>*> wf_out__,
                      int                                        j0__,
                      int                                        n__)
{
    transform<T>(spla_ctx__ , ispn__, 1.0, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, wf_out__, j0__, n__);
}

inline void
transform(::spla::Context& spla_ctx__, int ispn__, std::vector<Wave_functions<float>*> wf_in__,
          int i0__, int m__, dmatrix<std::complex<double>>& mtrx__, int irow0__, int jcol0__,
          std::vector<Wave_functions<float>*> wf_out__, int j0__, int n__)
{
    transform(spla_ctx__ , ispn__, 1.0, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, wf_out__, j0__, n__);
}

inline void
transform(::spla::Context& spla_ctx__, int ispn__, std::vector<Wave_functions<float>*> wf_in__,
          int i0__, int m__, dmatrix<double>& mtrx__, int irow0__, int jcol0__,
          std::vector<Wave_functions<float>*> wf_out__, int j0__, int n__)
{
    transform(spla_ctx__ , ispn__, 1.0, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, wf_out__, j0__, n__);
}


template <typename T>
inline void transform(::spla::Context& spla_ctx__,
                      int                           ispn__,
                      Wave_functions<real_type<T>>& wf_in__,
                      int                           i0__,
                      int                           m__,
                      dmatrix<T>&                   mtrx__,
                      int                           irow0__,
                      int                           jcol0__,
                      Wave_functions<real_type<T>>& wf_out__,
                      int                           j0__,
                      int                           n__)
{
    transform<T>(spla_ctx__, ispn__, 1.0, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, {&wf_out__}, j0__, n__);
}

inline void
transform(::spla::Context& spla_ctx__, int ispn__, Wave_functions<float>& wf_in__, int i0__, int m__,
          dmatrix<std::complex<double>>& mtrx__, int irow0__, int jcol0__,
          Wave_functions<float>& wf_out__, int j0__, int n__)
{
    transform(spla_ctx__, ispn__, 1.0, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, {&wf_out__}, j0__, n__);
}

inline void
transform(::spla::Context& spla_ctx__, int ispn__, Wave_functions<float>& wf_in__, int i0__, int m__,
          dmatrix<double>& mtrx__, int irow0__, int jcol0__,
          Wave_functions<float>& wf_out__, int j0__, int n__)
{
    transform(spla_ctx__, ispn__, 1.0, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, 0.0, {&wf_out__}, j0__, n__);
}


}
#endif

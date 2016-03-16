// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file orthogonalize.cpp
 *   
 *  \brief Contains implementation of sirius::Band::orthogonalize() method.
 */

#include "band.h"

namespace sirius {

template <>
void Band::orthogonalize<double_complex>(K_point* kp__,
                                         int N__,
                                         int n__,
                                         Wave_functions<false>& phi__,
                                         Wave_functions<false>& hphi__,
                                         Wave_functions<false>& ophi__,
                                         matrix<double_complex>& o__) const
{
    return;
}

template <>
void Band::orthogonalize<double>(K_point* kp__,
                                 int N__,
                                 int n__,
                                 Wave_functions<false>& phi__,
                                 Wave_functions<false>& hphi__,
                                 Wave_functions<false>& ophi__,
                                 matrix<double>& o__) const
{
    PROFILE_WITH_TIMER("sirius::Band::orthogonalize");

    if (N__ > 0)
    {
        /* project out the old subspace */
        phi__.inner<double>(0, N__, ophi__, N__, n__, o__, 0, 0); 

        linalg<CPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__, -1.0, (double*)&phi__(0, 0), 2 * kp__->num_gkvec_loc(),
                          &o__(0, 0), o__.ld(), 1.0, (double*)&phi__(0, N__), 2 * kp__->num_gkvec_loc());

        linalg<CPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__, -1.0, (double*)&hphi__(0, 0), 2 * kp__->num_gkvec_loc(),
                          &o__(0, 0), o__.ld(), 1.0, (double*)&hphi__(0, N__), 2 * kp__->num_gkvec_loc());

        linalg<CPU>::gemm(0, 0, 2 * kp__->num_gkvec_loc(), n__, N__, -1.0, (double*)&ophi__(0, 0), 2 * kp__->num_gkvec_loc(),
                          &o__(0, 0), o__.ld(), 1.0, (double*)&ophi__(0, N__), 2 * kp__->num_gkvec_loc());
    }

    /* orthogonalize new n__ x n__ block */
    phi__.inner<double>(N__, n__, ophi__, N__, n__, o__, 0, 0);
    
    int info;
    if ((info = linalg<CPU>::potrf(n__, &o__(0, 0), o__.ld())))
    {
        std::stringstream s;
        s << "error in factorization, info = " << info;
        TERMINATE(s);
    }

    if (linalg<CPU>::trtri(n__, &o__(0, 0), o__.ld()))
        TERMINATE("error in inversion");

    linalg<CPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, 1.0, &o__(0, 0), o__.ld(), (double*)&phi__(0, N__), 2 * kp__->num_gkvec_loc());
    linalg<CPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, 1.0, &o__(0, 0), o__.ld(), (double*)&hphi__(0, N__), 2 * kp__->num_gkvec_loc());
    linalg<CPU>::trmm('R', 'U', 'N', 2 * kp__->num_gkvec_loc(), n__, 1.0, &o__(0, 0), o__.ld(), (double*)&ophi__(0, N__), 2 * kp__->num_gkvec_loc());

    //// --== DEBUG ==--
    //phi__.inner<double>(0, N__ + n__, ophi__, 0, N__ + n__, o__, 0, 0);
    //for (int i = 0; i < N__ + n__; i++)
    //{
    //    for (int j = 0; j < N__ + n__; j++)
    //    {
    //        double a = o__(j, i);
    //        if (i == j) a -= 1;

    //        if (std::abs(a) > 1e-10)
    //        {
    //            printf("wrong overlap");
    //            TERMINATE("wrong overlap");
    //        }
    //    }
    //}
}

}

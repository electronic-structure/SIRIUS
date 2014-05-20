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

/** \file force.h
 *   
 *  \brief Contains definition of sirius::Force class.
 */

#ifndef __FORCE_H__
#define __FORCE_H__

#include "global.h"
#include "k_point.h"
#include "band.h"
#include "potential.h"
#include "density.h"
#include "mdarray.h"

namespace sirius
{

/// Compute Hellmann–Feynman forces.
class Force
{
    private:

        /** In the second-variational approach we need to compute the following expression for the k-dependent 
         *  contribution to the forces:
         *  \f[
         *      {\bf F}_{\rm IBS}^{\alpha}=\sum_{\bf k}w_{\bf k}\sum_{l\sigma}n_{l{\bf k}}
         *      \sum_{ij}c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
         *      {\bf F}_{ij}^{\alpha{\bf k}}
         *  \f]
         *  First, we sum over band and spin indices to get the "density matrix":
         *  \f[
         *      q_{ij} = \sum_{l\sigma}n_{l{\bf k}} c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
         *  \f]
         */
        static void compute_dmat(Global& parameters_, K_point* kp, mdarray<double_complex, 2>& dm);

        static void ibs_force(Global& parameters_, Band* band, K_point* kp, mdarray<double, 2>& ffac, mdarray<double, 2>& force);

    public:

        static void total_force(Global& parameters_, Potential* potential, Density* density, K_set* ks, mdarray<double, 2>& force);
};

}

#endif // __FORCE_H__

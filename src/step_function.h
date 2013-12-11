// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __STEP_FUNCTION_H__
#define __STEP_FUNCTION_H__

/** \file step_function.h

    \brief Step function for the full potential methods: 1 in the interstitial and 0 inside muffin-tin spheres.
*/
namespace sirius {

class Step_function
{
    private:

        Unit_cell* unit_cell_;

        Reciprocal_lattice* reciprocal_lattice_;
    
        /// plane wave expansion coefficients of the step function
        std::vector<complex16> step_function_pw_;
        
        /// step function on the real-space grid
        std::vector<double> step_function_;
        
        void init();

    public:

        Step_function(Unit_cell* unit_cell__, Reciprocal_lattice* reciprocal_lattice__) : 
            unit_cell_(unit_cell__), reciprocal_lattice_(reciprocal_lattice__)
        {
            init();
        }

        void get_step_function_form_factors(mdarray<double, 2>& ffac);
        
        inline complex16 theta_pw(int ig)
        {
            return step_function_pw_[ig];
        }

        inline double theta_it(int ir)
        {
            return step_function_[ir];
        }
};

#include "step_function.hpp"

};

#endif //  __STEP_FUNCTION_H__

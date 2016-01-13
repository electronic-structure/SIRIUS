// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file beta_projectors.h
 *   
 *  \brief Contains declaration and implementation of sirius::Beta_projectors class.
 */

#ifndef __BETA_PROJECTORS_H__
#define __BETA_PROJECTORS_H__

#include "gpu.h"
#include "communicator.h"
#include "unit_cell.h"
#include "wave_functions.h"
#include "sbessel.h"

#ifdef __GPU
extern "C" void create_beta_gk_gpu(int num_atoms,
                                   int num_gkvec,
                                   int const* beta_desc,
                                   cuDoubleComplex const* beta_gk_t,
                                   double const* gkvec,
                                   double const* atom_pos,
                                   cuDoubleComplex* beta_gk);
#endif

namespace sirius {

class Beta_projectors
{
    private:

        Communicator const& comm_;

        Unit_cell const& unit_cell_;

        Gvec const& gkvec_;

        mdarray<double, 2> gkvec_coord_;

        int lmax_beta_;

        processing_unit_t pu_;

        int num_gkvec_loc_;

        /// Total number of beta-projectors among atom types.
        int num_beta_t_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for atoms.
        matrix<double_complex> beta_gk_;

        mdarray<double_complex, 1> beta_phi_;

        struct beta_chunk
        {
            int num_beta_;
            int num_atoms_;
            int offset_;
            mdarray<int, 2> desc_;
            mdarray<double, 2> atom_pos_;
        };

        std::vector<beta_chunk> beta_chunks_;

        int max_num_beta_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t();
                    
        void split_in_chunks();

    public:

        Beta_projectors(Communicator const& comm__,
                        Unit_cell const& unit_cell__,
                        Gvec const& gkvec__,
                        processing_unit_t pu__)
            : comm_(comm__),
              unit_cell_(unit_cell__),
              gkvec_(gkvec__),
              lmax_beta_(unit_cell_.lmax()),
              pu_(pu__)
        {
            num_gkvec_loc_ = gkvec_.num_gvec(comm_.rank());

            split_in_chunks();

            generate_beta_gk_t();

            #ifdef __GPU
            if (pu_ == GPU)
            {
                gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc_);
                /* copy G+k vectors */
                for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++)
                {
                    int igk = gkvec_.offset_gvec(comm_.rank()) + igk_loc;
                    auto gc = gkvec_.gvec_shifted(igk);
                    for (auto x: {0, 1, 2}) gkvec_coord_(x, igk_loc) = gc[x];
                }
                gkvec_coord_.allocate_on_device();
                gkvec_coord_.copy_to_device();

                beta_gk_t_.allocate_on_device();
                beta_gk_t_.copy_to_device();
            }
            #endif

            beta_gk_ = matrix<double_complex>(num_gkvec_loc_, max_num_beta_);
        }

        matrix<double_complex>& beta_gk_t()
        {
            return beta_gk_t_;
        }

        matrix<double_complex> const& beta_gk() const
        {
            return beta_gk_;
        }

        mdarray<double_complex, 1> const& beta_phi() const
        {
            return beta_phi_;
        }

        Unit_cell const& unit_cell() const
        {
            return unit_cell_;
        }

        inline int num_beta_chunks() const
        {
            return static_cast<int>(beta_chunks_.size());
        }

        inline beta_chunk const& beta_chunk(int idx__) const
        {
            return beta_chunks_[idx__];
        }

        inline int num_gkvec_loc() const
        {
            return num_gkvec_loc_;
        }

        void generate(int chunk__);

        void inner(int chunk__, Wave_functions<false>& phi__, int idx0__, int n__);

        void prepare()
        {
            #ifdef __GPU
            if (pu_ == GPU)
            {
                beta_gk_.allocate_on_device();
                beta_phi_.allocate_on_device();
            }
            #endif
        }

        void dismiss()
        {
            #ifdef __GPU
            if (pu_ == GPU)
            {
                beta_gk_.deallocate_on_device();
                beta_phi_.deallocate_on_device();
            }
            #endif
        }
};

};

#endif

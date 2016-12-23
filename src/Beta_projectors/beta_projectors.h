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

/** \file beta_projectors.h
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors class.
 */

#ifndef __BETA_PROJECTORS_H__
#define __BETA_PROJECTORS_H__

#include "gpu.h"
#include "communicator.hpp"
#include "unit_cell.h"
#include "wave_functions.hpp"
#include "sbessel.h"
#include "simulation_context.h"

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

enum beta_desc_idx {
    nbf      = 0,
    offset   = 1,
    offset_t = 2,
    ia       = 3
};

class Beta_projectors_gradient;

/// Stores <G+k | beta> expansion
class Beta_projectors
{
    friend class Beta_projectors_gradient;

    protected:

        Communicator const& comm_;

        Unit_cell const& unit_cell_;

        Gvec const& gkvec_;

        mdarray<double, 2> gkvec_coord_;

        int lmax_beta_;

        device_t pu_;

        int num_gkvec_loc_;

        /// Total number of beta-projectors among atom types.
        int num_beta_t_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for all atoms.
        matrix<double_complex> beta_gk_a_;

        /// Plane-wave coefficients of |beta> functions for a chunk of atoms.
        matrix<double_complex> beta_gk_;
        
        /// Inner product between beta-projectors and wave-functions.
        /** Store as double to handle both gamma- and general k-point cases */
        mdarray<double, 1> beta_phi_;

        #ifdef __GPU
        /// Explicit GPU buffer for beta-projectors.
        matrix<double_complex> beta_gk_gpu_;
        #endif

        struct beta_chunk_t
        {
            int num_beta_;
            int num_atoms_;
            int offset_;
            mdarray<int, 2> desc_;
            mdarray<double, 2> atom_pos_;
        };

        mdarray<beta_chunk_t, 1> beta_chunks_;

        int max_num_beta_;

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t(Simulation_context const& ctx__);
                    
        void split_in_chunks();

        /// calculates < Beta | Psi > inner product
        template <typename T>
        void inner(int chunk__,
                   wave_functions& phi__,
                   int idx0__,
                   int n__,
                   mdarray<double_complex, 2>& beta_gk__,
                   mdarray<double, 1>& beta_phi__);

    public:

        Beta_projectors(Simulation_context const& ctx__,
                        Communicator const& comm__,
                        Gvec const& gkvec__);

        matrix<double_complex>& beta_gk_t()
        {
            return beta_gk_t_;
        }

        matrix<double_complex> const& beta_gk_a()
        {
            return beta_gk_a_;
        }

        matrix<double_complex> const& beta_gk() const
        {
            return beta_gk_;
        }

        template <typename T>
        matrix<T> beta_phi(int chunk__, int n__)
        {
            int nbeta = beta_chunk(chunk__).num_beta_;
            if (pu_ == GPU) {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()),
                                           reinterpret_cast<T*>(beta_phi_.at<GPU>()),
                                           nbeta, n__));
            } else {
                return std::move(matrix<T>(reinterpret_cast<T*>(beta_phi_.at<CPU>()),
                                           nbeta, n__));
            }
        }

        Unit_cell const& unit_cell() const
        {
            return unit_cell_;
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        Gvec const& gk_vectors() const
        {
            return gkvec_;
        }

        device_t proc_unit() const
        {
            return pu_;
        }

        int lmax_beta() const
        {
            return lmax_beta_;
        }

        inline int num_beta_chunks() const
        {
            return static_cast<int>(beta_chunks_.size());
        }

        inline beta_chunk_t const& beta_chunk(int idx__) const
        {
            return beta_chunks_(idx__);
        }

        inline int num_gkvec_loc() const
        {
            return num_gkvec_loc_;
        }

        void generate(int chunk__);

        template <typename T>
        void inner(int chunk__, wave_functions& phi__, int idx0__, int n__)
        {
            inner<T>(chunk__, phi__, idx0__, n__, beta_gk_, beta_phi_);
        }

        int max_num_beta()
        {
            return max_num_beta_;
        }

        void prepare()
        {
            #ifdef __GPU
            if (pu_ == GPU) {
                beta_gk_gpu_.allocate(memory_t::device);
                beta_phi_.allocate(memory_t::device);
            }
            #endif
        }

        void dismiss()
        {
            #ifdef __GPU
            if (pu_ == GPU) {
                beta_gk_gpu_.deallocate_on_device();
                beta_phi_.deallocate_on_device();
            }
            #endif
        }
};

};

#endif

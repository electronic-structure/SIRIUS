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

/** \file generate_d_operator_matrix.h
 *   
 *  \brief Contains implementation of sirius::Potential::generate_D_operator_matrix method.
 */

#include "potential.h"

namespace sirius {

#ifdef __GPU
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__,
                                                double_complex const* veff__,
                                                int const* gvec__,
                                                double const* atom_pos__,
                                                double_complex* veff_a__);
#endif

void Potential::generate_D_operator_matrix()
{
    PROFILE_WITH_TIMER("sirius::Potential::generate_D_operator_matrix");

    if (!parameters_.esm_type() == ultrasoft_pseudopotential) STOP(); // decide what to do in this case

    std::vector<Periodic_function<double>*> veff_vec(parameters_.num_mag_dims() + 1);
    veff_vec[0] = effective_potential_;
    for (int j = 0; j < parameters_.num_mag_dims(); j++) veff_vec[1 + j] = effective_magnetic_field_[j];

    auto rl = ctx_.reciprocal_lattice();

    #ifdef __GPU
    mdarray<int, 2> gvec;
    if (parameters_.processing_unit() == GPU)
    {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
             auto type = unit_cell_.atom_type(iat);
             type->uspp().q_pw.allocate_on_device();
             type->uspp().q_pw.copy_to_device();
        }
    
        gvec = mdarray<int, 2>(3, spl_num_gvec_.local_size());
        for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
        {
            for (int x: {0, 1, 2}) gvec(x, igloc) = ctx_.gvec()[spl_num_gvec_[igloc]][x];
        }
        gvec.allocate_on_device();
        gvec.copy_to_device();
    }
    #endif

    for (int iv = 0; iv < parameters_.num_mag_dims() + 1; iv++)
    {
        /* get plane-wave coefficients of effective potential */
        veff_vec[iv]->fft_transform(-1);

        #ifdef __GPU
        mdarray<double_complex, 1> veff;
        if (parameters_.processing_unit() == GPU)
        {
            veff = mdarray<double_complex, 1>(&veff_vec[iv]->f_pw(spl_num_gvec_.global_offset()), spl_num_gvec_.local_size());
            veff.allocate_on_device();
            veff.copy_to_device();
        }
        #endif

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            auto atom_type = unit_cell_.atom_type(iat);
            int nbf = atom_type->mt_basis_size();
            matrix<double_complex> d_tmp(nbf * (nbf + 1) / 2, atom_type->num_atoms()); 

            if (parameters_.processing_unit() == CPU)
            {
                matrix<double_complex> veff_a(spl_num_gvec_.local_size(), atom_type->num_atoms());

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < atom_type->num_atoms(); i++)
                {
                    int ia = atom_type->atom_id(i);

                    for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
                    {
                        int ig = spl_num_gvec_[igloc];
                        veff_a(igloc, i) = veff_vec[iv]->f_pw(ig) * rl->gvec_phase_factor(ig, ia);
                    }
                }

                linalg<CPU>::gemm(2, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), spl_num_gvec_.local_size(),
                                  atom_type->uspp().q_pw.at<CPU>(), spl_num_gvec_.local_size(),
                                  veff_a.at<CPU>(), spl_num_gvec_.local_size(), d_tmp.at<CPU>(), d_tmp.ld());
            }
            #ifdef __GPU
            if (parameters_.processing_unit() == GPU)
            {
                matrix<double_complex> veff_a(nullptr, spl_num_gvec_.local_size(), atom_type->num_atoms());
                veff_a.allocate_on_device();
                
                d_tmp.allocate_on_device();

                mdarray<double, 2> atom_pos(3, atom_type->num_atoms());
                for (int i = 0; i < atom_type->num_atoms(); i++)
                {
                    int ia = atom_type->atom_id(i);
                    for (int x: {0, 1, 2}) atom_pos(x, i) = unit_cell_.atom(ia)->position(x);
                }
                atom_pos.allocate_on_device();
                atom_pos.copy_to_device();

                mul_veff_with_phase_factors_gpu(atom_type->num_atoms(),
                                                spl_num_gvec_.local_size(),
                                                veff.at<GPU>(),
                                                gvec.at<GPU>(),
                                                atom_pos.at<GPU>(),
                                                veff_a.at<GPU>());

                linalg<GPU>::gemm(2, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), spl_num_gvec_.local_size(),
                                  atom_type->uspp().q_pw.at<GPU>(), spl_num_gvec_.local_size(),
                                  veff_a.at<GPU>(), spl_num_gvec_.local_size(), d_tmp.at<GPU>(), d_tmp.ld());

                d_tmp.copy_to_host();
            }
            #endif

            comm_.allreduce(d_tmp.at<CPU>(), static_cast<int>(d_tmp.size()));

            #ifdef __PRINT_OBJECT_CHECKSUM
            auto z = d_tmp.checksum();
            DUMP("checksum(d_mtrx): %18.10f %18.10f", z.real(), z.imag());
            #endif

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type->num_atoms(); i++)
            {
                int ia = atom_type->atom_id(i);

                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    for (int xi1 = 0; xi1 <= xi2; xi1++)
                    {
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

                        if (xi1 == xi2)
                        {
                            assert(std::abs(d_tmp(idx12, i).imag()) < 1e-10);
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2, iv) = d_tmp(idx12, i).real() * unit_cell_.omega();
                        }
                        else
                        {
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2, iv) = d_tmp(idx12, i) * unit_cell_.omega();
                            unit_cell_.atom(ia)->d_mtrx(xi2, xi1, iv) = std::conj(d_tmp(idx12, i)) * unit_cell_.omega();
                        }
                    }
                }
            }
        }
    }

    /* add d_ion to the effective potential component of D-operator */
    #pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        auto atom_type = unit_cell_.atom(ia)->type();
        int nbf = unit_cell_.atom(ia)->mt_basis_size();

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = atom_type->indexb(xi2).lm;
            int idxrf2 = atom_type->indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                int lm1 = atom_type->indexb(xi1).lm;
                int idxrf1 = atom_type->indexb(xi1).idxrf;
                
                if (lm1 == lm2) unit_cell_.atom(ia)->d_mtrx(xi1, xi2, 0) += atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
            }
        }
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
             auto type = unit_cell_.atom_type(iat);
             type->uspp().q_pw.deallocate_on_device();
        }
    }
    #endif
}

};

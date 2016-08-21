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

/** \file augment.cpp
 *   
 *  \brief Contains implementation of sirius::Density::augment function.
 */

#include "density.h"

namespace sirius {

#ifdef __GPU
extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int num_beta__,
                                   double const* atom_pos__,
                                   int const* gvec__,
                                   double const* dm__,
                                   double* dm_pw__);

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double_complex const* q_pw_t__,
                                   double const* dm_pw__,
                                   double_complex* rho_pw__);
#endif

// TODO: looks like only real part of Q(G) is sufficient; x2 factor in copying to GPU
//       check if this also holds for the non-collinear case

void Density::augment(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::augment");

    /* split G-vectors between ranks */
    splindex<block> spl_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());
    
    /* collect density and magnetization into single array */
    std::vector<Periodic_function<double>*> rho_vec(ctx_.num_mag_dims() + 1);
    rho_vec[0] = rho_;
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        rho_vec[1 + j] = magnetization_[j];
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    for (auto e: rho_vec)
    {
        auto cs = e->checksum_pw();
        DUMP("checksum(rho_vec_pw): %20.14f %20.14f", cs.real(), cs.imag());
    }
    #endif

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
        ctx_.augmentation_op(iat).prepare(0);

    #ifdef __GPU
    mdarray<int, 2> gvec;
    mdarray<double_complex, 2> rho_pw_gpu;
    if (ctx_.processing_unit() == GPU)
    {
        gvec = mdarray<int, 2>(3, spl_gvec.local_size());
        for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
        {
            for (int x: {0, 1, 2}) gvec(x, igloc) = ctx_.gvec().gvec(spl_gvec[igloc])[x];
        }
        gvec.allocate_on_device();
        gvec.copy_to_device();
        
        rho_pw_gpu = mdarray<double_complex, 2>(spl_gvec.local_size(), ctx_.num_mag_dims() + 1);
        rho_pw_gpu.allocate_on_device();
        rho_pw_gpu.zero_on_device();
    }
    #endif

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto& atom_type = unit_cell_.atom_type(iat);
        if (!atom_type.uspp().augmentation_) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();
        
        /* convert to real matrix */
        mdarray<double, 3> dm(nbf * (nbf + 1) / 2, atom_type.num_atoms(), ctx_.num_mag_dims() + 1);
        #pragma omp parallel for
        for (int i = 0; i < atom_type.num_atoms(); i++)
        {
            int ia = atom_type.atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    switch (ctx_.num_mag_dims())
                    {
                        case 0:
                        {
                            dm(idx12, i, 0) = density_matrix_(xi2, xi1, 0, ia).real();
                            break;
                        }
                        case 1:
                        {
                            dm(idx12, i, 0) = std::real(density_matrix_(xi2, xi1, 0, ia) + density_matrix_(xi2, xi1, 1, ia));
                            dm(idx12, i, 1) = std::real(density_matrix_(xi2, xi1, 0, ia) - density_matrix_(xi2, xi1, 1, ia));
                            break;
                        }
                    }
                }
            }
        }

        if (ctx_.processing_unit() == CPU)
        {
            runtime::Timer t2("sirius::Density::augment|phase_fac");
            /* treat phase factors as real array with x2 size */
            mdarray<double, 2> phase_factors(atom_type.num_atoms(), spl_gvec.local_size() * 2);

            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
            {
                int ig = spl_gvec[igloc];
                for (int i = 0; i < atom_type.num_atoms(); i++)
                {
                    int ia = atom_type.atom_id(i);
                    double_complex z = std::conj(ctx_.gvec_phase_factor(ig, ia));
                    phase_factors(i, 2 * igloc)     = z.real();
                    phase_factors(i, 2 * igloc + 1) = z.imag();
                }
            }
            t2.stop();
            
            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2);

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
            {
                runtime::Timer t3("sirius::Density::augment|gemm");
                linalg<CPU>::gemm(0, 0, nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2, atom_type.num_atoms(), 
                                  &dm(0, 0, iv), dm.ld(),
                                  &phase_factors(0, 0), phase_factors.ld(), 
                                  &dm_pw(0, 0), dm_pw.ld());
                t3.stop();

                #ifdef __PRINT_OBJECT_CHECKSUM
                {
                    auto cs = dm_pw.checksum();
                    ctx_.comm().allreduce(&cs, 1);
                    DUMP("checksum(dm_pw) : %18.10f", cs);
                }
                #endif

                runtime::Timer t4("sirius::Density::augment|sum");
                #pragma omp parallel for
                for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
                {
                    //double ar = 0;
                    //double ai = 0;
                    double_complex zsum(0, 0);
                    /* get contribution from non-diagonal terms */
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++)
                    {
                        double_complex z1 = ctx_.augmentation_op(iat).q_pw(i, igloc);
                        //double q = 2.0 * ctx_.augmentation_op(iat).q_pw(i, igloc).real();
                        double_complex z2(dm_pw(i, 2 * igloc), dm_pw(i, 2 * igloc + 1));

                        /* D_{xi2,xi1} * Q(G)_{xi1, xi2} + D_{xi1,xi2} * Q(G)_{xix, xi1}^{+} */
                        //ar += dm_pw(i, 2 * igloc)     * q;
                        //ai += dm_pw(i, 2 * igloc + 1) * q;
                        zsum += z1 * z2 * 2.0;
                    }
                    /* remove one diagonal contribution which was double-counted */
                    for (int xi = 0; xi < nbf; xi++)
                    {
                        int i = xi * (xi + 1) / 2 + xi;
                        double_complex z1 = ctx_.augmentation_op(iat).q_pw(i, igloc);
                        double_complex z2(dm_pw(i, 2 * igloc), dm_pw(i, 2 * igloc + 1));
                        zsum -= z1 * z2;

                        //double q = ctx_.augmentation_op(iat).q_pw(i, igloc).real();
                        //ar -= dm_pw(i, 2 * igloc)     * q;
                        //ai -= dm_pw(i, 2 * igloc + 1) * q;

                    }
                    rho_vec[iv]->f_pw(spl_gvec[igloc]) += zsum; //double_complex(ar, ai);
                }
                t4.stop();
            }
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU)
        {
            mdarray<double, 2> atom_pos(3, atom_type.num_atoms());
            #pragma omp parallel for
            for (int i = 0; i < atom_type.num_atoms(); i++)
            {
                int ia = atom_type.atom_id(i);
                auto pos = unit_cell_.atom(ia).position();
                for (int x: {0, 1, 2}) atom_pos(x, i) = pos[x];
            }
            atom_pos.allocate_on_device();
            atom_pos.copy_to_device();

            dm.allocate_on_device();
            dm.copy_to_device();

            /* treat auxiliary array as double with x2 size */
            mdarray<double, 2> dm_pw(nullptr, nbf * (nbf + 1) / 2, spl_gvec.local_size() * 2);
            dm_pw.allocate_on_device();

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
            {
                generate_dm_pw_gpu(atom_type.num_atoms(),
                                   spl_gvec.local_size(),
                                   nbf,
                                   atom_pos.at<GPU>(),
                                   gvec.at<GPU>(),
                                   dm.at<GPU>(0, 0, iv),
                                   dm_pw.at<GPU>());
                
                sum_q_pw_dm_pw_gpu(spl_gvec.local_size(), 
                                   nbf,
                                   ctx_.augmentation_op(iat).q_pw().at<GPU>(),
                                   dm_pw.at<GPU>(),
                                   rho_pw_gpu.at<GPU>(0, iv));
            }
        }
        #endif
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        rho_pw_gpu.copy_to_host();
        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
        {
            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
            {
                int ig = spl_gvec[igloc];
                rho_vec[iv]->f_pw(ig) += rho_pw_gpu(igloc, iv);
            }
        }
    }
    #endif

    runtime::Timer t5("sirius::Density::augment|mpi");
    for (auto e: rho_vec)
    {
        ctx_.comm().allgather(&e->f_pw(0), spl_gvec.global_offset(), spl_gvec.local_size());

        #ifdef __PRINT_OBJECT_CHECKSUM
        {
            auto cs = e->checksum_pw();
            DUMP("checksum(rho_vec_pw): %20.14f %20.14f", cs.real(), cs.imag());
        }
        #endif
    }
    t5.stop();

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
        ctx_.augmentation_op(iat).dismiss(0);
}

};


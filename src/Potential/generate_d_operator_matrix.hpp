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

/** \file generate_d_operator_matrix.hpp
 *   
 *  \brief Contains implementation of sirius::Potential::generate_D_operator_matrix method.
 */

#ifdef __GPU
extern "C" void mul_veff_with_phase_factors_gpu(int                   num_atoms__,
                                                int                   num_gvec_loc__,
                                                double_complex const* veff__,
                                                int const*            gvec__,
                                                double const*         atom_pos__,
                                                double*               veff_a__,
                                                int                   stream_id__);
#endif

inline void Potential::generate_D_operator_matrix()
{
    PROFILE("sirius::Potential::generate_D_operator_matrix");

    /* store effective potential and magnetic field in a vector */
    std::vector<Periodic_function<double>*> veff_vec(ctx_.num_mag_dims() + 1);
    veff_vec[0] = effective_potential_.get();
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        veff_vec[1 + j] = effective_magnetic_field_[j];
    }
   
    #ifdef __GPU
    mdarray<double_complex, 1> veff_tmp(nullptr, ctx_.gvec().count());
    if (ctx_.processing_unit() == GPU) {
        veff_tmp.allocate(memory_t::device);
    }
    #endif
    
    ctx_.augmentation_op(0).prepare(0);

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type.mt_basis_size();
        
        /* start copy of Q(G) for the next atom type */
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            acc::sync_stream(0);
            if (iat + 1 != unit_cell_.num_atom_types()) {
                ctx_.augmentation_op(iat + 1).prepare(0);
            }
        }
        #endif
        
        /* trivial case */
        if (!atom_type.pp_desc().augment) {
            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    int ia = atom_type.atom_id(i);
                    auto& atom = unit_cell_.atom(ia);

                    for (int xi2 = 0; xi2 < nbf; xi2++) {
                        for (int xi1 = 0; xi1 < nbf; xi1++) {
                            atom.d_mtrx(xi1, xi2, iv) = 0;
                        }
                    }
                }
            }
            continue;
        }
        matrix<double> d_tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms()); 
        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
            switch (ctx_.processing_unit()) {
                case CPU: {
                    matrix<double> veff_a(2 * ctx_.gvec().count(), atom_type.num_atoms());

                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        int ia = atom_type.atom_id(i);

                        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                            int ig = ctx_.gvec().offset() + igloc;
                            /* V(G) * exp(i * G * r_{alpha}) */
                            auto z = veff_vec[iv]->f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia);
                            veff_a(2 * igloc,     i) = z.real();
                            veff_a(2 * igloc + 1, i) = z.imag();
                        }
                    }

                    linalg<CPU>::gemm(0, 0, nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                      ctx_.augmentation_op(iat).q_pw(), veff_a, d_tmp);
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    /* copy plane wave coefficients of effective potential to GPU */
                    mdarray<double_complex, 1> veff(&veff_vec[iv]->f_pw_local(0), veff_tmp.at<GPU>(),
                                                    ctx_.gvec().count());
                    veff.copy<memory_t::host, memory_t::device>();

                    matrix<double> veff_a(2 * ctx_.gvec().count(), atom_type.num_atoms(), memory_t::device);
                    
                    d_tmp.allocate(memory_t::device);

                    mul_veff_with_phase_factors_gpu(atom_type.num_atoms(),
                                                    ctx_.gvec().count(),
                                                    veff.at<GPU>(),
                                                    ctx_.gvec_coord().at<GPU>(),
                                                    ctx_.atom_coord(iat).at<GPU>(),
                                                    veff_a.at<GPU>(), 1);

                    linalg<GPU>::gemm(0, 0, nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                      ctx_.augmentation_op(iat).q_pw(), veff_a, d_tmp, 1);

                    d_tmp.copy<memory_t::device, memory_t::host>();
                    #endif
                    break;
                }
            }

            if (ctx_.gvec().reduced()) {
                if (comm_.rank() == 0) {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nbf * (nbf + 1) / 2; j++) {
                            d_tmp(j, i) = 2 * d_tmp(j, i) - veff_vec[iv]->f_pw_local(0).real() * ctx_.augmentation_op(iat).q_pw(j, 0);
                        }
                    }
                } else {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nbf * (nbf + 1) / 2; j++) {
                            d_tmp(j, i) *= 2;
                        }
                    }
                }
            }

            comm_.allreduce(d_tmp.at<CPU>(), static_cast<int>(d_tmp.size()));

            if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
                auto cs = d_tmp.checksum();
                print_checksum("D-op matrix of valence", cs);
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type.num_atoms(); i++) {
                int ia = atom_type.atom_id(i);
                auto& atom = unit_cell_.atom(ia);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* D-matix is symmetric */
                        atom.d_mtrx(xi1, xi2, iv) = atom.d_mtrx(xi2, xi1, iv) = d_tmp(idx12, i) * unit_cell_.omega();
                    }
                }
            }
        }
        ctx_.augmentation_op(iat).dismiss();
    }

    /* add d_ion to the effective potential component of D-operator */
    #pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom_type = unit_cell_.atom(ia).type();
        int nbf = unit_cell_.atom(ia).mt_basis_size();

        for (int xi2 = 0; xi2 < nbf; xi2++) {
            int lm2 = atom_type.indexb(xi2).lm;
            int idxrf2 = atom_type.indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++) {
                int lm1 = atom_type.indexb(xi1).lm;
                int idxrf1 = atom_type.indexb(xi1).idxrf;

                if (lm1 == lm2) {
                    unit_cell_.atom(ia).d_mtrx(xi1, xi2, 0) += atom_type.pp_desc().d_mtrx_ion(idxrf1, idxrf2);
                }
            }
        }
    }
    
    if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto cs = unit_cell_.atom(ia).d_mtrx().checksum();
            std::stringstream s;
            s << "atom_" << ia;
            print_checksum(s.str(), cs);
        }
    }
}

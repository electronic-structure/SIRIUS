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

void Potential::generate_d_mtrx()
{
    PROFILE();

    Timer t("sirius::Potential::generate_d_mtrx");

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        /* get plane-wave coefficients of effective potential */
        effective_potential_->fft_transform(-1);

        auto rl = ctx_.reciprocal_lattice();

        #ifdef __GPU
        mdarray<double_complex, 1> veff;
        mdarray<int, 2> gvec;

        if (parameters_.processing_unit() == GPU)
        {
            veff = mdarray<double_complex, 1>(&effective_potential_->f_pw((int)spl_num_gvec_.global_offset()), 
                                              spl_num_gvec_.local_size());
            veff.allocate_on_device();
            veff.copy_to_device();

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                 auto type = unit_cell_.atom_type(iat);
                 type->uspp().q_pw.allocate_on_device();
                 type->uspp().q_pw.copy_to_device();
            }
        
            gvec = mdarray<int, 2>(3, spl_num_gvec_.local_size());
            for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
            {
                for (int x = 0; x < 3; x++) gvec(x, igloc) = ctx_.gvec()[igloc][x];
            }
            gvec.allocate_on_device();
            gvec.copy_to_device();
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

                    for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
                    {
                        int ig = (int)spl_num_gvec_[igloc];
                        veff_a(igloc, i) = effective_potential_->f_pw(ig) * std::conj(rl->gvec_phase_factor(ig, ia));
                    }
                }

                linalg<CPU>::gemm(1, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), (int)spl_num_gvec_.local_size(),
                                  &atom_type->uspp().q_pw(0, 0), (int)spl_num_gvec_.local_size(),
                                  &veff_a(0, 0), (int)spl_num_gvec_.local_size(),
                                  &d_tmp(0, 0), d_tmp.ld());
            }
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef __GPU
                matrix<double_complex> veff_a(nullptr, spl_num_gvec_.local_size(), atom_type->num_atoms());
                veff_a.allocate_on_device();
                
                d_tmp.allocate_on_device();

                mdarray<double, 2> atom_pos(3, atom_type->num_atoms());
                for (int i = 0; i < atom_type->num_atoms(); i++)
                {
                    int ia = atom_type->atom_id(i);
                    for (int x = 0; x < 3; x++) atom_pos(x, i) = unit_cell_.atom(ia)->position(x);
                }
                atom_pos.allocate_on_device();
                atom_pos.copy_to_device();

                mul_veff_with_phase_factors_gpu(atom_type->num_atoms(),
                                                (int)spl_num_gvec_.local_size(),
                                                veff.at<GPU>(),
                                                gvec.at<GPU>(),
                                                atom_pos.at<GPU>(),
                                                veff_a.at<GPU>());

                linalg<GPU>::gemm(1, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), (int)spl_num_gvec_.local_size(),
                                  atom_type->uspp().q_pw.at<GPU>(), (int)spl_num_gvec_.local_size(),
                                  veff_a.at<GPU>(), (int)spl_num_gvec_.local_size(), d_tmp.at<GPU>(), d_tmp.ld());

                d_tmp.copy_to_host();
                #else
                TERMINATE_NO_GPU
                #endif
            }

            comm_.allreduce(d_tmp.at<CPU>(), (int)d_tmp.size());

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type->num_atoms(); i++)
            {
                int ia = atom_type->atom_id(i);

                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    int lm2 = atom_type->indexb(xi2).lm;
                    int idxrf2 = atom_type->indexb(xi2).idxrf;
                    for (int xi1 = 0; xi1 <= xi2; xi1++)
                    {
                        int lm1 = atom_type->indexb(xi1).lm;
                        int idxrf1 = atom_type->indexb(xi1).idxrf;
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

                        if (xi1 == xi2)
                        {
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2) = real(d_tmp(idx12, i)) * unit_cell_.omega() +
                                                             atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
                        }
                        else
                        {
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2) = d_tmp(idx12, i) * unit_cell_.omega();
                            unit_cell_.atom(ia)->d_mtrx(xi2, xi1) = conj(d_tmp(idx12, i)) * unit_cell_.omega();
                            if (lm1 == lm2)
                            {
                                unit_cell_.atom(ia)->d_mtrx(xi1, xi2) += atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
                                unit_cell_.atom(ia)->d_mtrx(xi2, xi1) += atom_type->uspp().d_mtrx_ion(idxrf2, idxrf1);
                            }
                        }
                    }
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
}

};

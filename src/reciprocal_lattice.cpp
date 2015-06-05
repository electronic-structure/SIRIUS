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

/** \file reciprocal_lattice.cpp
 *
 *  \brief Contains remaining implementation of sirius::Reciprocal_lattice class. 
 */

#include "reciprocal_lattice.h"

namespace sirius {
        
Reciprocal_lattice::Reciprocal_lattice(Unit_cell const& unit_cell__, 
                                       electronic_structure_method_t esm_type__,
                                       FFT3D<CPU>* fft__,
                                       int lmax__,
                                       Communicator const& comm__)
    : unit_cell_(unit_cell__), 
      esm_type_(esm_type__),
      fft_(fft__),
      comm_(comm__)
{
    reciprocal_lattice_vectors_ = unit_cell_.reciprocal_lattice_vectors();
    inverse_reciprocal_lattice_vectors_ = inverse(reciprocal_lattice_vectors_);

    init(lmax__);
}

Reciprocal_lattice::~Reciprocal_lattice()
{
}

void Reciprocal_lattice::init(int lmax)
{
    Timer t("sirius::Reciprocal_lattice::init");
    
    /* create split index */
    spl_num_gvec_ = splindex<block>(num_gvec(), comm_.size(), comm_.rank());
    
    if (lmax >= 0)
    {
        /* precompute spherical harmonics of G-vectors */
        gvec_ylm_ = mdarray<double_complex, 2>(Utils::lmmax(lmax), spl_num_gvec_.local_size());
        
        Timer t2("sirius::Reciprocal_lattice::init|ylm_G");
        for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
        {
            int ig = (int)spl_num_gvec_[igloc];
            auto rtp = SHT::spherical_coordinates(gvec_cart(ig));
            SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_ylm_(0, igloc));
        }
        t2.stop();
    }
    
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        int nbeta = unit_cell_.max_mt_radial_basis_size();

        mdarray<double, 4> q_radial_functions(unit_cell_.max_num_mt_points(), lmax + 1, nbeta * (nbeta + 1) / 2, 
                                              unit_cell_.num_atom_types());

        fix_q_radial_functions(q_radial_functions);

        // TODO: in principle, this can be distributed over G-shells (each mpi rank holds radial integrals only for
        //       G-shells of local fraction of G-vectors
        mdarray<double, 4> q_radial_integrals(nbeta * (nbeta + 1) / 2, lmax + 1, unit_cell_.num_atom_types(), 
                                              num_gvec_shells_inner());

        generate_q_radial_integrals(lmax, q_radial_functions, q_radial_integrals);

        generate_q_pw(lmax, q_radial_integrals);
    }

    /* precompute G-vector phase factors */
    #ifdef __CACHE_GVEC_PHASE_FACTORS
    gvec_phase_factors_ = mdarray<double_complex, 2>(spl_num_gvec_.local_size(), unit_cell_.num_atoms());
    #pragma omp parallel for
    for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
    {
        int ig = (int)spl_num_gvec_[igloc];
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) gvec_phase_factors_(igloc, ia) = gvec_phase_factor<global>(ig, ia);
    }
    #endif
}

std::vector<double_complex> Reciprocal_lattice::make_periodic_function(mdarray<double, 2>& form_factors, int ngv) const
{
    assert((int)form_factors.size(0) == unit_cell_.num_atom_types());
    
    std::vector<double_complex> f_pw(ngv, double_complex(0, 0));

    double fourpi_omega = fourpi / unit_cell_.omega();

    splindex<block> spl_ngv(ngv, comm_.size(), comm_.rank());

    #pragma omp parallel
    for (auto it = splindex_iterator<block>(spl_ngv); it.valid(); it++)
    {
        int ig = (int)it.idx();
        int igs = gvec_shell(ig);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {            
            int iat = unit_cell_.atom(ia)->type_id();
            f_pw[ig] += fourpi_omega * conj(gvec_phase_factor<global>(ig, ia)) * form_factors(iat, igs);
        }
    }

    comm_.allgather(&f_pw[0], (int)spl_ngv.global_offset(), (int)spl_ngv.local_size());

    return f_pw;
}


void Reciprocal_lattice::fix_q_radial_functions(mdarray<double, 4>& qrf)
{
    Timer t("sirius::Reciprocal_lattice::fix_q_radial_functions");

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
        {
            for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
            {
                for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                {
                    int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                    memcpy(&qrf(0, l3, idx, iat), &atom_type->uspp().q_radial_functions(0, idx), 
                           atom_type->num_mt_points() * sizeof(double));
                    atom_type->fix_q_radial_function(l3, idxrf1, idxrf2, &qrf(0, l3, idx, iat));
                }
            }
        }
    }
}

void Reciprocal_lattice::generate_q_radial_integrals(int lmax, mdarray<double, 4>& qrf, mdarray<double, 4>& qri)
{
    Timer t("sirius::Reciprocal_lattice::generate_q_radial_integrals");

    qri.zero();
    
    splindex<block> spl_num_gvec_shells(num_gvec_shells_inner(), comm_.size(), comm_.rank());
    
    #pragma omp parallel
    {
        sbessel_pw<double> jl(unit_cell_, lmax);
        for (auto it = splindex_iterator<block>(spl_num_gvec_shells); it.valid(); it++)
        {
            int igs = (int)it.idx();
            jl.load(gvec_shell_len(igs));

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_.atom_type(iat);
                Spline<double> s(atom_type->radial_grid());

                for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
                {
                    for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
                    {
                        int l2 = atom_type->indexr(idxrf2).l;
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                        {
                            int l1 = atom_type->indexr(idxrf1).l;

                            int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                            
                            if (l3 >= abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0)
                            {
                                for (int ir = 0; ir < atom_type->num_mt_points(); ir++)
                                    s[ir] = jl(ir, l3, iat) * qrf(ir, l3, idx, iat);

                                qri(idx, l3, iat, igs) = s.interpolate().integrate(0);
                            }
                        }
                    }
                }
            }
        }
    }
    int ld = (int)(qri.size(0) * qri.size(1) * qri.size(2));
    comm_.allgather(&qri(0, 0, 0, 0), ld * (int)spl_num_gvec_shells.global_offset(), ld * (int)spl_num_gvec_shells.local_size());
}

void Reciprocal_lattice::generate_q_pw(int lmax, mdarray<double, 4>& qri)
{
    Timer t("sirius::Reciprocal_lattice::generate_q_pw");

    double fourpi_omega = fourpi / unit_cell_.omega();
    
    std::vector<int> l_by_lm = Utils::l_by_lm(lmax);

    std::vector<double_complex> zilm(Utils::lmmax(lmax));
    for (int l = 0, lm = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++, lm++) zilm[lm] = pow(double_complex(0, 1), l);
    }

    mdarray<double, 2> gvec_rlm(Utils::lmmax(lmax), spl_num_gvec_.local_size());
    for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
    {
        int ig = (int)spl_num_gvec_[igloc];
        auto rtp = SHT::spherical_coordinates(gvec_cart(ig));
        SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_rlm(0, igloc));
    }

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type->mt_basis_size();
        int lmax_beta = atom_type->indexr().lmax();
        int lmmax = Utils::lmmax(lmax_beta * 2);
        Gaunt_coefficients<double> gaunt_coefs(lmax_beta, 2 * lmax_beta, lmax_beta, SHT::gaunt_rlm);

        atom_type->uspp().q_mtrx.zero();
        
        atom_type->uspp().q_pw = mdarray<double_complex, 2>(spl_num_gvec_.local_size(), nbf * (nbf + 1) / 2);

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = atom_type->indexb(xi2).lm;
            int idxrf2 = atom_type->indexb(xi2).idxrf;

            for (int xi1 = 0; xi1 <= xi2; xi1++)
            {
                int lm1 = atom_type->indexb(xi1).lm;
                int idxrf1 = atom_type->indexb(xi1).idxrf;

                int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                int idxrf12 = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                
                #pragma omp parallel
                {
                    std::vector<double_complex> v(lmmax);
                    for (auto it = splindex_iterator<block>(spl_num_gvec_); it.valid(); it++)
                    {
                        int igs = gvec_shell((int)it.idx());
                        int igloc = (int)it.idx_local();
                        for (int lm3 = 0; lm3 < lmmax; lm3++)
                        {
                            v[lm3] = conj(zilm[lm3]) * gvec_rlm(lm3, igloc) * qri(idxrf12, l_by_lm[lm3], iat, igs);
                        }

                        atom_type->uspp().q_pw(igloc, idx12) = fourpi_omega * gaunt_coefs.sum_L3_gaunt(lm2, lm1, &v[0]);

                        if (igs == 0)
                        {
                            atom_type->uspp().q_mtrx(xi1, xi2) = unit_cell_.omega() * atom_type->uspp().q_pw(0, idx12);
                            atom_type->uspp().q_mtrx(xi2, xi1) = std::conj(atom_type->uspp().q_mtrx(xi1, xi2));
                        }
                    }
                }
            }
        }
        comm_.bcast(&atom_type->uspp().q_mtrx(0, 0), (int)atom_type->uspp().q_mtrx.size(), 0);
    }
}

}

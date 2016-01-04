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
// TODO: get rid of this class

#include "reciprocal_lattice.h"
#include "debug.hpp"

namespace sirius {
        
Reciprocal_lattice::Reciprocal_lattice(Unit_cell const& unit_cell__, 
                                       electronic_structure_method_t esm_type__,
                                       Gvec const& gvec__,
                                       int lmax__,
                                       Communicator const& comm__)
    : unit_cell_(unit_cell__), 
      esm_type_(esm_type__),
      gvec_(gvec__),
      comm_(comm__)
{
    PROFILE();

    init(lmax__);
}

Reciprocal_lattice::~Reciprocal_lattice()
{
}

void Reciprocal_lattice::init(int lmax)
{
    PROFILE();

    Timer t("sirius::Reciprocal_lattice::init");
    
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        int nbeta = unit_cell_.max_mt_radial_basis_size();

        mdarray<double, 4> q_radial_functions(unit_cell_.max_num_mt_points(), lmax + 1, nbeta * (nbeta + 1) / 2, 
                                              unit_cell_.num_atom_types());

        fix_q_radial_functions(q_radial_functions);

        // TODO: in principle, this can be distributed over G-shells (each mpi rank holds radial integrals only for
        //       G-shells of local fraction of G-vectors
        mdarray<double, 4> q_radial_integrals(nbeta * (nbeta + 1) / 2, lmax + 1, unit_cell_.num_atom_types(), 
                                              gvec_.num_shells());

        generate_q_radial_integrals(lmax, q_radial_functions, q_radial_integrals);

        generate_q_pw(lmax, q_radial_integrals);
    }
}


void Reciprocal_lattice::fix_q_radial_functions(mdarray<double, 4>& qrf)
{
    PROFILE();

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
                    memcpy(&qrf(0, l3, idx, iat), &atom_type->uspp().q_radial_functions_l(0, idx, l3), 
                           atom_type->num_mt_points() * sizeof(double));
                    
                    if (atom_type->uspp().num_q_coefs)
                        atom_type->fix_q_radial_function(l3, idxrf1, idxrf2, &qrf(0, l3, idx, iat));
                }
            }
        }
    }
}

void Reciprocal_lattice::generate_q_radial_integrals(int lmax, mdarray<double, 4>& qrf, mdarray<double, 4>& qri)
{
    PROFILE();

    Timer t("sirius::Reciprocal_lattice::generate_q_radial_integrals");

    qri.zero();
    
    splindex<block> spl_num_gvec_shells(gvec_.num_shells(), comm_.size(), comm_.rank());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        int nrf = atom_type->mt_radial_basis_size();

        mdarray<Spline<double>, 2> qrf_spline(2 * atom_type->indexr().lmax() + 1, nrf * (nrf + 1) / 2);
        
        for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
        {
            #pragma omp parallel for
            for (int idx = 0; idx < nrf * (nrf + 1) / 2; idx++)
            {
                qrf_spline(l3, idx) = Spline<double>(atom_type->radial_grid());

                for (int ir = 0; ir < atom_type->num_mt_points(); ir++)
                    qrf_spline(l3, idx)[ir] = qrf(ir, l3, idx, iat);
                qrf_spline(l3, idx).interpolate();
            }
        }
        #pragma omp parallel for
        for (int ishloc = 0; ishloc < (int)spl_num_gvec_shells.local_size(); ishloc++)
        {
            int igs = (int)spl_num_gvec_shells[ishloc];
            Spherical_Bessel_functions jl(2 * atom_type->indexr().lmax(), atom_type->radial_grid(), gvec_.shell_len(igs));

            for (int l3 = 0; l3 <= 2 * atom_type->indexr().lmax(); l3++)
            {
                for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
                {
                    int l2 = atom_type->indexr(idxrf2).l;
                    for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                    {
                        int l1 = atom_type->indexr(idxrf1).l;

                        int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                        
                        if (l3 >= std::abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0)
                        {
                            qri(idx, l3, iat, igs) = inner(jl(l3), qrf_spline(l3, idx), 0, atom_type->num_mt_points());
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
    PROFILE();

    Timer t("sirius::Reciprocal_lattice::generate_q_pw");

    double fourpi_omega = fourpi / unit_cell_.omega();
    
    std::vector<int> l_by_lm = Utils::l_by_lm(lmax);

    std::vector<double_complex> zilm(Utils::lmmax(lmax));
    for (int l = 0, lm = 0; l <= lmax; l++)
    {
        for (int m = -l; m <= l; m++, lm++) zilm[lm] = std::pow(double_complex(0, 1), l);
    }
    
    splindex<block> spl_num_gvec(gvec_.num_gvec(), comm_.size(), comm_.rank());
    mdarray<double, 2> gvec_rlm(Utils::lmmax(lmax), spl_num_gvec.local_size());
    for (int igloc = 0; igloc < spl_num_gvec.local_size(); igloc++)
    {
        int ig = spl_num_gvec[igloc];
        auto rtp = SHT::spherical_coordinates(gvec_.cart(ig));
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
        
        atom_type->uspp().q_pw = mdarray<double_complex, 2>(spl_num_gvec.local_size(), nbf * (nbf + 1) / 2);

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
                    #pragma omp for
                    for (int igloc = 0; igloc < spl_num_gvec.local_size(); igloc++)
                    {
                        int ig = spl_num_gvec[igloc];
                        int igs = gvec_.shell(ig);
                        for (int lm3 = 0; lm3 < lmmax; lm3++)
                        {
                            v[lm3] = std::conj(zilm[lm3]) * gvec_rlm(lm3, igloc) * qri(idxrf12, l_by_lm[lm3], iat, igs);
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
        #ifdef __PRINT_OBJECT_CHECKSUM
        auto z = atom_type->uspp().q_pw.checksum();
        DUMP("checksum(Q(G)) : %18.10f %18.10f", std::real(z), std::imag(z));
        #endif
    }
}

}

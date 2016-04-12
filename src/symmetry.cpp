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

/** \file symmetry.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Symmetry class.
 */

#include "symmetry.h"
#include "sht.h"

namespace sirius {

Symmetry::Symmetry(matrix3d<double>& lattice_vectors__,  
                   int num_atoms__,
                   mdarray<double, 2>& positions__,
                   mdarray<double, 2>& spins__,
                   std::vector<int>& types__,
                   double tolerance__)
    : lattice_vectors_(lattice_vectors__),
      num_atoms_(num_atoms__),
      types_(types__),
      tolerance_(tolerance__)
{
    PROFILE();

    if (lattice_vectors__.det() < 0)
    {
        std::stringstream s;
        s << "spglib requires positive determinant for a matrix of lattice vectors";
        TERMINATE(s);
    }

    double lattice[3][3];
    for (int i: {0, 1, 2})
    {
        for (int j: {0, 1, 2}) lattice[i][j] = lattice_vectors_(i, j);
    }
    positions_ = mdarray<double, 2>(3, num_atoms_);
    for (int ia = 0; ia < num_atoms_; ia++)
    {
        for (int x: {0, 1, 2}) positions_(x, ia) = positions__(x, ia);
    }

    spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions_(0, 0), &types_[0], num_atoms_, tolerance_);
    if (spg_dataset_ == NULL)
    {
        TERMINATE("spg_get_dataset() returned NULL");
    }

    if (spg_dataset_->spacegroup_number == 0)
        TERMINATE("spg_get_dataset() returned 0 for the space group");

    if (spg_dataset_->n_atoms != num_atoms__)
    {
        std::stringstream s;
        s << "spg_get_dataset() returned wrong number of atoms (" << spg_dataset_->n_atoms << ")" << std::endl
          << "expected number of atoms is " <<  num_atoms__;
        TERMINATE(s);
    }

    inverse_lattice_vectors_ = inverse(lattice_vectors_);

    for (int isym = 0; isym < spg_dataset_->n_operations; isym++)
    {
        space_group_symmetry_descriptor sym_op;

        sym_op.R = matrix3d<int>(spg_dataset_->rotations[isym]);
        sym_op.t = vector3d<double>(spg_dataset_->translations[isym][0],
                                    spg_dataset_->translations[isym][1],
                                    spg_dataset_->translations[isym][2]);
        int p = sym_op.R.det(); 
        if (!(p == 1 || p == -1)) TERMINATE("wrong rotation matrix");
        sym_op.proper = p;
        sym_op.rotation = lattice_vectors_ * matrix3d<double>(sym_op.R * p) * inverse_lattice_vectors_;
        sym_op.euler_angles = euler_angles(sym_op.rotation);

        space_group_symmetry_.push_back(sym_op);
    }

    sym_table_ = mdarray<int, 2>(num_atoms_, num_spg_sym());
    /* loop over spatial symmetries */
    for (int isym = 0; isym < num_spg_sym(); isym++)
    {
        for (int ia = 0; ia < num_atoms_; ia++)
        {
            auto R = space_group_symmetry(isym).R;
            auto t = space_group_symmetry(isym).t;
            /* spatial transform */
            vector3d<double> pos(positions__(0, ia), positions__(1, ia), positions__(2, ia));
            auto v = Utils::reduce_coordinates(R * pos + t);

            int ja = -1;
            /* check for equivalent atom */
            for (int k = 0; k < num_atoms_; k++)
            {
                vector3d<double> pos1(positions__(0, k), positions__(1, k), positions__(2, k));
                if ((v.first - pos1).length() < 1e-6)
                {
                    ja = k;
                    break;
                }
            }

            if (ja == -1) TERMINATE("equivalent atom not found");
            sym_table_(ia, isym) = ja;
        }
    }
    
    /* loop over spatial symmetries */
    for (int isym = 0; isym < num_spg_sym(); isym++)
    {
        /* loop over spin symmetries */
        for (int jsym = 0; jsym < num_spg_sym(); jsym++)
        {
            /* take proper part of rotation matrix */
            auto Rspin = space_group_symmetry(jsym).rotation;
            
            int n = 0;
            /* check if all atoms transfrom under spatial and spin symmetries */
            for (int ia = 0; ia < num_atoms_; ia++)
            {
                int ja = sym_table_(ia, isym);

                /* now check tha vector filed transforms from atom ia to atom ja */
                /* vector field of atom is expected to be in Cartesian coordinates */
                auto vd = Rspin * vector3d<double>(spins__(0, ia), spins__(1, ia), spins__(2, ia)) -
                          vector3d<double>(spins__(0, ja), spins__(1, ja), spins__(2, ja));

                if (vd.length() < 1e-10) n++;
            }
            /* if all atoms transform under spin rotaion, add it to a list */
            if (n == num_atoms_)
            {
                magnetic_group_symmetry_descriptor mag_op;
                mag_op.spg_op = space_group_symmetry(isym);
                mag_op.isym = isym;
                mag_op.spin_rotation = Rspin;
                magnetic_group_symmetry_.push_back(mag_op);
                break;
            }
        }
    }
}

Symmetry::~Symmetry()
{
    spg_free_dataset(spg_dataset_);
}

matrix3d<double> Symmetry::rot_mtrx_cart(vector3d<double> euler_angles) const
{
    double alpha = euler_angles[0];
    double beta = euler_angles[1];
    double gamma = euler_angles[2];

    matrix3d<double> rm;
    rm(0, 0) = std::cos(alpha) * std::cos(beta) * std::cos(gamma) - std::sin(alpha) * std::sin(gamma);
    rm(0, 1) = -std::cos(gamma) * std::sin(alpha) - std::cos(alpha) * std::cos(beta) * std::sin(gamma);
    rm(0, 2) = std::cos(alpha) * std::sin(beta);
    rm(1, 0) = std::cos(beta) * std::cos(gamma) * std::sin(alpha) + std::cos(alpha) * std::sin(gamma);
    rm(1, 1) = std::cos(alpha) * std::cos(gamma) - std::cos(beta) * std::sin(alpha) * std::sin(gamma);
    rm(1, 2) = std::sin(alpha) * std::sin(beta);
    rm(2, 0) = -std::cos(gamma) * std::sin(beta);
    rm(2, 1) = std::sin(beta) * std::sin(gamma);
    rm(2, 2) = std::cos(beta);

    return rm;
}

vector3d<double> Symmetry::euler_angles(matrix3d<double> const& rot__) const
{
    vector3d<double> angles(0, 0, 0);
    
    if (std::abs(rot__.det() - 1) > 1e-10)
    {
        std::stringstream s;
        s << "determinant of rotation matrix is " << rot__.det();
        TERMINATE(s);
    }

    if (std::abs(rot__(2, 2) - 1.0) < 1e-10) // cos(beta) == 1, beta = 0
    {
        angles[0] = Utils::phi_by_sin_cos(rot__(1, 0), rot__(0, 0));
    }
    else if (std::abs(rot__(2, 2) + 1.0) < 1e-10) // cos(beta) == -1, beta = Pi
    {
        angles[0] = Utils::phi_by_sin_cos(-rot__(0, 1), rot__(1, 1));
        angles[1] = pi;
    }
    else             
    {
        double beta = std::acos(rot__(2, 2));
        angles[0] = Utils::phi_by_sin_cos(rot__(1, 2) / std::sin(beta), rot__(0, 2) / std::sin(beta));
        angles[1] = beta;
        angles[2] = Utils::phi_by_sin_cos(rot__(2, 1) / std::sin(beta), -rot__(2, 0) / std::sin(beta));
    }

    auto rm1 = rot_mtrx_cart(angles);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (std::abs(rot__(i, j) - rm1(i, j)) > 1e-10)
            {
                std::stringstream s;
                s << "matrices don't match" << std::endl
                  << "initial symmetry matrix: " << std::endl
                  << rot__(0, 0) << " " << rot__(0, 1) << " " << rot__(0, 2) << std::endl
                  << rot__(1, 0) << " " << rot__(1, 1) << " " << rot__(1, 2) << std::endl
                  << rot__(2, 0) << " " << rot__(2, 1) << " " << rot__(2, 2) << std::endl
                  << "euler angles : " << angles[0] / pi << " " << angles[1] / pi << " " << angles[2] / pi << std::endl
                  << "computed symmetry matrix : " << std::endl
                  << rm1(0, 0) << " " << rm1(0, 1) << " " << rm1(0, 2) << std::endl
                  << rm1(1, 0) << " " << rm1(1, 1) << " " << rm1(1, 2) << std::endl
                  << rm1(2, 0) << " " << rm1(2, 1) << " " << rm1(2, 2) << std::endl;
                TERMINATE(s);
            }
        }
    }

    return angles;
}

int Symmetry::get_irreducible_reciprocal_mesh(vector3d<int> k_mesh__,
                                              vector3d<int> is_shift__,
                                              mdarray<double, 2>& kp__,
                                              std::vector<double>& wk__) const
{
    int nktot = k_mesh__[0] * k_mesh__[1] * k_mesh__[2];

    mdarray<int, 2> grid_address(3, nktot);
    std::vector<int> ikmap(nktot);

    double lattice[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) lattice[i][j] = lattice_vectors_(i, j);
    }

    int nknr = spg_get_ir_reciprocal_mesh((int(*)[3])&grid_address(0, 0),
                                          &ikmap[0],
                                          &k_mesh__[0],
                                          &is_shift__[0], 1, 
                                          lattice,
                                          (double(*)[3])&positions_(0, 0),
                                          &types_[0],
                                          num_atoms_,
                                          tolerance_);

    std::map<int, int> wknr;
    for (int ik = 0; ik < nktot; ik++)
    {
        if (wknr.count(ikmap[ik]) == 0) wknr[ikmap[ik]] = 0;
        wknr[ikmap[ik]] += 1;
    }

    wk__ = std::vector<double>(nknr);
    kp__ = mdarray<double, 2>(3, nknr);

    int n = 0;
    for (auto it = wknr.begin(); it != wknr.end(); it++)
    {
        wk__[n] = double(it->second) / nktot;
        for (int x = 0; x < 3; x++) kp__(x, n) = double(grid_address(x, it->first) + is_shift__[x] / 2.0) / k_mesh__[x];
        n++;
    }

    return nknr;
}

void Symmetry::check_gvec_symmetry(Gvec const& gvec__) const
{
    PROFILE();

    for (int isym = 0; isym < num_mag_sym(); isym++)
    {
        auto sm = magnetic_group_symmetry(isym).spg_op.R;

        for (int ig = 0; ig < gvec__.num_gvec(); ig++)
        {
            auto gv = gvec__[ig];
            /* apply symmetry operation to the G-vector */
            auto gv_rot = transpose(sm) * gv;
            /* check limits */
            for (int x = 0; x < 3; x++)
            {
                auto limits = gvec__.fft_box().limits(x);
                /* check boundaries */
                if (gv_rot[x] < limits.first || gv_rot[x] > limits.second)
                {
                    std::stringstream s;
                    s << "rotated G-vector is outside of grid limits" << std::endl
                      << "original G-vector: " << gv << std::endl
                      << "rotation matrix: " << std::endl
                      << sm(0, 0) << " " << sm(0, 1) << " " << sm(0, 2) << std::endl
                      << sm(1, 0) << " " << sm(1, 1) << " " << sm(1, 2) << std::endl
                      << sm(2, 0) << " " << sm(2, 1) << " " << sm(2, 2) << std::endl
                      << "rotated G-vector: " << gv_rot;
                      TERMINATE(s);
                }
            }
            int ig_rot = gvec__.index_by_gvec(gv_rot);
            /* special case where -G is equal to G */
            if (ig_rot == -1 && gvec__.reduced())
            {
                gv_rot = gv_rot * (-1);
                ig_rot = gvec__.index_by_gvec(gv_rot);
            }
            if (ig_rot < 0 || ig_rot >= gvec__.num_gvec())
            {
                std::stringstream s;
                s << "rotated G-vector index is wrong" << std::endl
                  << "original G-vector: " << gv << std::endl
                  << "rotation matrix: " << std::endl
                  << sm(0, 0) << " " << sm(0, 1) << " " << sm(0, 2) << std::endl
                  << sm(1, 0) << " " << sm(1, 1) << " " << sm(1, 2) << std::endl
                  << sm(2, 0) << " " << sm(2, 1) << " " << sm(2, 2) << std::endl
                  << "rotated G-vector: " << gv_rot << std::endl
                  << "rotated G-vector index: " << ig_rot << std::endl
                  << "number of G-vectors: " << gvec__.num_gvec();
                  TERMINATE(s);
            }
        }
    }
}

void Symmetry::symmetrize_function(double_complex* f_pw__,
                                   Gvec const& gvec__,
                                   Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_function", comm__);

    splindex<block> spl_gvec(gvec__.num_gvec(), comm__.size(), comm__.rank());
    mdarray<double_complex, 1> sym_f_pw(gvec__.num_gvec());
    sym_f_pw.zero();
    
    double* ptr = (double*)&sym_f_pw(0);

    #pragma omp parallel for
    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        auto R = magnetic_group_symmetry(i).spg_op.R;
        auto t = magnetic_group_symmetry(i).spg_op.t;

        for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
        {
            int ig = spl_gvec[igloc];
            
            double_complex z = f_pw__[ig] * std::exp(double_complex(0, twopi * (gvec__[ig] * t)));

            /* apply symmetry operation to the G-vector;
             * remember that we move R from acting on x to acting on G: G(Rx) = (GR)x;
             * GR is a vector-matrix multiplication [G][.....]
             *                                         [..R..]
             *                                         [.....]
             * which can also be written as matrix^{T}-vector operation
             */
            auto gv_rot = transpose(R) * gvec__[ig];

            /* index of a rotated G-vector */
            int ig_rot = gvec__.index_by_gvec(gv_rot);

            if (gvec__.reduced() && ig_rot == -1)
            {
                gv_rot = gv_rot * (-1);
                int ig_rot = gvec__.index_by_gvec(gv_rot);
              
                #pragma omp atomic update
                ptr[2 * ig_rot] += z.real();

                #pragma omp atomic update
                ptr[2 * ig_rot + 1] -= z.imag();
            }
            else
            {
                assert(ig_rot >= 0 && ig_rot < gvec__.num_gvec());
              
                #pragma omp atomic update
                ptr[2 * ig_rot] += z.real();

                #pragma omp atomic update
                ptr[2 * ig_rot + 1] += z.imag();
            }
        }
    }
    comm__.allreduce(&sym_f_pw(0), gvec__.num_gvec());
    
    double nrm = 1 / double(num_mag_sym());
    #pragma omp parallel for
    for (int ig = 0; ig < gvec__.num_gvec(); ig++) f_pw__[ig] = sym_f_pw(ig) * nrm;
}

void Symmetry::symmetrize_vector_z_component(double_complex* f_pw__,
                                             Gvec const& gvec__,
                                             Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_vector_z_component");
    
    splindex<block> spl_gvec(gvec__.num_gvec(), comm__.size(), comm__.rank());
    mdarray<double_complex, 1> sym_f_pw(gvec__.num_gvec());
    sym_f_pw.zero();

    double* ptr = (double*)&sym_f_pw(0);

    #pragma omp parallel for
    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        auto R = magnetic_group_symmetry(i).spg_op.R;
        auto t = magnetic_group_symmetry(i).spg_op.t;
        auto S = magnetic_group_symmetry(i).spin_rotation;

        for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
        {
            int ig = spl_gvec[igloc];

            auto gv_rot = transpose(R) * gvec__[ig];

            /* index of a rotated G-vector */
            int ig_rot = gvec__.index_by_gvec(gv_rot);

            assert(ig_rot >= 0 && ig_rot < gvec__.num_gvec());

            double_complex z = f_pw__[ig] * std::exp(double_complex(0, twopi * (gvec__[ig] * t))) * S(2, 2);
            
            #pragma omp atomic update
            ptr[2 * ig_rot] += real(z);

            #pragma omp atomic update
            ptr[2 * ig_rot + 1] += imag(z);
        }
    }
    comm__.allreduce(&sym_f_pw(0), gvec__.num_gvec());

    for (int ig = 0; ig < gvec__.num_gvec(); ig++) f_pw__[ig] = sym_f_pw(ig) / double(num_mag_sym());

}

void Symmetry::symmetrize_vector(double_complex* fx_pw__,
                                 double_complex* fy_pw__,
                                 double_complex* fz_pw__,
                                 Gvec const& gvec__,
                                 Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_vector");
    
    splindex<block> spl_gvec(gvec__.num_gvec(), comm__.size(), comm__.rank());
    mdarray<double_complex, 1> sym_fx_pw(gvec__.num_gvec());
    mdarray<double_complex, 1> sym_fy_pw(gvec__.num_gvec());
    mdarray<double_complex, 1> sym_fz_pw(gvec__.num_gvec());
    sym_fx_pw.zero();
    sym_fy_pw.zero();
    sym_fz_pw.zero();

    double* ptr_x = (double*)&sym_fx_pw(0);
    double* ptr_y = (double*)&sym_fy_pw(0);
    double* ptr_z = (double*)&sym_fz_pw(0);

    std::vector<double_complex*> v_pw_in({fx_pw__, fy_pw__, fz_pw__});

    #pragma omp parallel for
    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        auto R = magnetic_group_symmetry(i).spg_op.R;
        auto t = magnetic_group_symmetry(i).spg_op.t;
        auto S = magnetic_group_symmetry(i).spin_rotation;

        for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
        {
            int ig = spl_gvec[igloc];

            auto gv_rot = transpose(R) * gvec__[ig];

            /* index of a rotated G-vector */
            int ig_rot = gvec__.index_by_gvec(gv_rot);

            assert(ig_rot >= 0 && ig_rot < gvec__.num_gvec());

            double_complex phase = std::exp(double_complex(0, twopi * (gvec__[ig] * t)));
            double_complex vz[] = {double_complex(0, 0), double_complex(0, 0), double_complex(0, 0)};
            for (int j: {0, 1, 2})
                for (int k: {0, 1, 2})
                    vz[j] += phase * S(j, k) * v_pw_in[k][ig];

            #pragma omp atomic update
            ptr_x[2 * ig_rot] += vz[0].real();

            #pragma omp atomic update
            ptr_y[2 * ig_rot] += vz[1].real();

            #pragma omp atomic update
            ptr_z[2 * ig_rot] += vz[2].real();

            #pragma omp atomic update
            ptr_x[2 * ig_rot + 1] += vz[0].imag();
            
            #pragma omp atomic update
            ptr_y[2 * ig_rot + 1] += vz[1].imag();

            #pragma omp atomic update
            ptr_z[2 * ig_rot + 1] += vz[2].imag();
        }
    }
    comm__.allreduce(&sym_fx_pw(0), gvec__.num_gvec());
    comm__.allreduce(&sym_fy_pw(0), gvec__.num_gvec());
    comm__.allreduce(&sym_fz_pw(0), gvec__.num_gvec());

    for (int ig = 0; ig < gvec__.num_gvec(); ig++)
    {
        fx_pw__[ig] = sym_fx_pw(ig) / double(num_mag_sym());
        fy_pw__[ig] = sym_fy_pw(ig) / double(num_mag_sym());
        fz_pw__[ig] = sym_fz_pw(ig) / double(num_mag_sym());
    }
}

void Symmetry::symmetrize_function(mdarray<double, 3>& frlm__,
                                   Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_function_mt");

    int lmmax = (int)frlm__.size(0);
    int nrmax = (int)frlm__.size(1);
    if (num_atoms_ != (int)frlm__.size(2)) TERMINATE("wrong number of atoms");

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 3> fsym(lmmax, nrmax, spl_atoms.local_size());
    fsym.zero();

    double alpha = 1.0 / double(num_mag_sym());

    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++)
        {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.second == comm__.rank())
            {
                linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha, rotm.at<CPU>(), rotm.ld(), 
                                  frlm__.at<CPU>(0, 0, ia), frlm__.ld(), 1.0,
                                  fsym.at<CPU>(0, 0, location.first), fsym.ld());
            }
        }
    }
    comm__.allgather(fsym.at<CPU>(), frlm__.at<CPU>(), 
                     lmmax * nrmax * spl_atoms.global_offset(), 
                     lmmax * nrmax * spl_atoms.local_size());
}

void Symmetry::symmetrize_vector_z_component(mdarray<double, 3>& vz_rlm__,
                                             Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_vector_z_component_mt");

    int lmmax = (int)vz_rlm__.size(0);
    int nrmax = (int)vz_rlm__.size(1);

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    if (num_atoms_ != (int)vz_rlm__.size(2)) TERMINATE("wrong number of atoms");

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 3> fsym(lmmax, nrmax, spl_atoms.local_size());
    fsym.zero();

    double alpha = 1.0 / double(num_mag_sym());

    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        auto S = magnetic_group_symmetry(i).spin_rotation;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++)
        {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.second == comm__.rank())
            {
                linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha * S(2, 2), rotm.at<CPU>(), rotm.ld(), 
                                  vz_rlm__.at<CPU>(0, 0, ia), vz_rlm__.ld(), 1.0,
                                  fsym.at<CPU>(0, 0, location.first), fsym.ld());
            }
        }
    }

    comm__.allgather(fsym.at<CPU>(), vz_rlm__.at<CPU>(), 
                     (int)(lmmax * nrmax * spl_atoms.global_offset()), 
                     (int)(lmmax * nrmax * spl_atoms.local_size()));
}

void Symmetry::symmetrize_vector(mdarray<double, 3>& vx_rlm__,
                                 mdarray<double, 3>& vy_rlm__,
                                 mdarray<double, 3>& vz_rlm__,
                                 Communicator const& comm__) const
{
    runtime::Timer t("sirius::Symmetry::symmetrize_vector_mt");

    int lmmax = (int)vx_rlm__.size(0);
    int nrmax = (int)vx_rlm__.size(1);

    splindex<block> spl_atoms(num_atoms_, comm__.size(), comm__.rank());

    int lmax = Utils::lmax_by_lmmax(lmmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    mdarray<double, 4> v_sym(lmmax, nrmax, spl_atoms.local_size(), 3);
    v_sym.zero();

    mdarray<double, 3> vtmp(lmmax, nrmax, 3);

    double alpha = 1.0 / double(num_mag_sym());

    std::vector<mdarray<double, 3>*> vrlm({&vx_rlm__, &vy_rlm__, &vz_rlm__});

    for (int i = 0; i < num_mag_sym(); i++)
    {
        /* full space-group symmetry operation is {R|t} */
        int pr = magnetic_group_symmetry(i).spg_op.proper;
        auto eang = magnetic_group_symmetry(i).spg_op.euler_angles;
        int isym = magnetic_group_symmetry(i).isym;
        auto S = magnetic_group_symmetry(i).spin_rotation;
        SHT::rotation_matrix(lmax, eang, pr, rotm);

        for (int ia = 0; ia < num_atoms_; ia++)
        {
            int ja = sym_table_(ia, isym);
            auto location = spl_atoms.location(ja);
            if (location.second == comm__.rank())
            {
                for (int k: {0, 1, 2}) 
                {
                    linalg<CPU>::gemm(0, 0, lmmax, nrmax, lmmax, alpha, rotm.at<CPU>(), rotm.ld(), 
                                      vrlm[k]->at<CPU>(0, 0, ia), vrlm[k]->ld(), 0.0,
                                      vtmp.at<CPU>(0, 0, k), vtmp.ld());
                }
                for (int k: {0, 1, 2})
                {
                    for (int j: {0, 1, 2})
                    {
                        for (int ir = 0; ir < nrmax; ir++)
                        {
                            for (int lm = 0; lm < lmmax; lm++)
                            {
                                v_sym(lm, ir, location.first, k) += S(k, j) * vtmp(lm, ir, j);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int k: {0, 1, 2})
    {
        comm__.allgather(v_sym.at<CPU>(0, 0, 0, k), vrlm[k]->at<CPU>(), 
                         (int)(lmmax * nrmax * spl_atoms.global_offset()), 
                         (int)(lmmax * nrmax * spl_atoms.local_size()));
    }
}

};

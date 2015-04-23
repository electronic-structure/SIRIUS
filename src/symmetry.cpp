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

/** \file symmetry.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Symmetry class.
 */

#include "symmetry.h"

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
    double lattice[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) lattice[i][j] = lattice_vectors_(i, j);
    }
    positions_ = mdarray<double, 2>(3, num_atoms_);
    positions__ >> positions_;

    spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions_(0, 0), &types_[0], num_atoms_, tolerance_);

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

    sym_table_ = mdarray<int, 2>(num_atoms_, num_sym_op());
    /* loop over spatial symmetries */
    for (int isym = 0; isym < num_sym_op(); isym++)
    {
        for (int ia = 0; ia < num_atoms_; ia++)
        {
            /* spatial transform */
            vector3d<double> pos(positions__(0, ia), positions__(1, ia), positions__(2, ia));
            auto v = Utils::reduce_coordinates(rot_mtrx(isym) * pos + fractional_translation(isym));

            int ja = -1;
            /* check for equivalent atom */
            for (int k = 0; k < num_atoms_; k++)
            {
                vector3d<double> pos1(positions__(0, k), positions__(1, k), positions__(2, k));
                if ((v.first - pos1).length() < 1e-10)
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
    for (int isym = 0; isym < num_sym_op(); isym++)
    {
        /* loop over spin symmetries */
        for (int jsym = 0; jsym < num_sym_op(); jsym++)
        {
            /* take proper part of rotation matrix */
            /* vector field of atom is expected to be in Cartesian coordinates */
            auto Rspin = rot_mtrx_cart(jsym);
            Rspin = Rspin * Rspin.det();
            
            int n = 0;
            /* check if all atoms transfrom under spatial and spin symmetries */
            for (int ia = 0; ia < num_atoms_; ia++)
            {
                int ja = sym_table_(ia, isym);

                /* now check tha vector filed transforms from atom ia to atom ja */
                auto vd = Rspin * vector3d<double>(spins__(0, ia), spins__(1, ia), spins__(2, ia)) -
                          vector3d<double>(spins__(0, ja), spins__(1, ja), spins__(2, ja));

                if (vd.length() < 1e-10) n++;
            }
            /* if all atoms transform under spin rotaion, add it to a list */
            if (n == num_atoms_)
            {
                mag_sym_.push_back(std::pair<int, int>(isym, jsym));
                break;
            }
        }
    }
}

Symmetry::~Symmetry()
{
    spg_free_dataset(spg_dataset_);
}

matrix3d<int> Symmetry::rot_mtrx(int isym__)
{
    return matrix3d<int>(spg_dataset_->rotations[isym__]);
}

matrix3d<double> Symmetry::rot_mtrx_cart(int isym__)
{
    double rot_mtrx_lat[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) rot_mtrx_lat[i][j] = spg_dataset_->rotations[isym__][i][j];
    }

    return lattice_vectors_ * matrix3d<double>(rot_mtrx_lat) * inverse_lattice_vectors_;
}


matrix3d<double> Symmetry::rot_mtrx(vector3d<double> euler_angles)
{
    double alpha = euler_angles[0];
    double beta = euler_angles[1];
    double gamma = euler_angles[2];

    matrix3d<double> rm;
    rm(0, 0) = cos(alpha) * cos(beta) * cos(gamma) - sin(alpha) * sin(gamma);
    rm(0, 1) = -cos(gamma) * sin(alpha) - cos(alpha) * cos(beta) * sin(gamma);
    rm(0, 2) = cos(alpha) * sin(beta);
    rm(1, 0) = cos(beta) * cos(gamma) * sin(alpha) + cos(alpha) * sin(gamma);
    rm(1, 1) = cos(alpha) * cos(gamma) - cos(beta) * sin(alpha) * sin(gamma);
    rm(1, 2) = sin(alpha) * sin(beta);
    rm(2, 0) = -cos(gamma) * sin(beta);
    rm(2, 1) = sin(beta) * sin(gamma);
    rm(2, 2) = cos(beta);

    return rm;
}

vector3d<double> Symmetry::euler_angles(int isym)
{
    const double eps = 1e-10;

    vector3d<double> angles(0.0);
    
    int p = proper_rotation(isym);

    auto rm = rot_mtrx(isym) * p;

    if (std::abs(rm(2, 2) - 1.0) < eps) // cos(beta) == 1, beta = 0
    {
        angles[0] = Utils::phi_by_sin_cos(rm(1, 0), rm(0, 0));
    }
    else if (std::abs(rm(2, 2) + 1.0) < eps) // cos(beta) == -1, beta = Pi
    {
        angles[0] = Utils::phi_by_sin_cos(-rm(0, 1), rm(1, 1));
        angles[1] = pi;
    }
    else             
    {
        double beta = acos(rm(2, 2));
        angles[0] = Utils::phi_by_sin_cos(rm(1, 2) / sin(beta), rm(0, 2) / sin(beta));
        angles[1] = beta;
        angles[2] = Utils::phi_by_sin_cos(rm(2, 1) / sin(beta), -rm(2, 0) / sin(beta));
    }

    auto rm1 = rot_mtrx(angles);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (std::abs(rm(i, j) - rm1(i, j)) > eps)
            {
                std::stringstream s;
                s << "matrices don't match for symmetry operation " << isym << std::endl
                  << "initial symmetry matrix : " << std::endl
                  << rm(0, 0) << " " << rm(0, 1) << " " << rm(0, 2) << std::endl
                  << rm(1, 0) << " " << rm(1, 1) << " " << rm(1, 2) << std::endl
                  << rm(2, 0) << " " << rm(2, 1) << " " << rm(2, 2) << std::endl
                  << "euler angles : " << angles[0] << " " << angles[1] << " " << angles[2] << std::endl
                  << "computed symmetry matrix : " << std::endl
                  << rm1(0, 0) << " " << rm1(0, 1) << " " << rm1(0, 2) << std::endl
                  << rm1(1, 0) << " " << rm1(1, 1) << " " << rm1(1, 2) << std::endl
                  << rm1(2, 0) << " " << rm1(2, 1) << " " << rm1(2, 2) << std::endl;
                error_local(__FILE__, __LINE__, s);
            }
        }
    }

    return angles;
}

int Symmetry::proper_rotation(int isym)
{
    matrix3d<int> rot_mtrx(spg_dataset_->rotations[isym]);
    int p = rot_mtrx.det();

    if (!(p == 1 || p == -1)) error_local(__FILE__, __LINE__, "wrong rotation matrix");

    return p;
}

int Symmetry::get_irreducible_reciprocal_mesh(vector3d<int> k_mesh__,
                                              vector3d<int> is_shift__,
                                              mdarray<double, 2>& kp__,
                                              std::vector<double>& wk__)
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

void Symmetry::check_gvec_symmetry(FFT3D<CPU>* fft__)
{
    for (int isym = 0; isym < num_sym_op(); isym++)
    {
        auto sm = rot_mtrx(isym);

        for (int ig = 0; ig < fft__->num_gvec(); ig++)
        {
            auto gv = fft__->gvec(ig);
            /* apply symmetry operation to the G-vector */
            vector3d<int> gv_rot = transpose(sm) * gv;
            for (int x = 0; x < 3; x++)
            {
                auto limits = fft__->grid_limits(x);
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
            int ig_rot = fft__->gvec_index(gv_rot);
            if (ig_rot >= fft__->num_gvec())
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
                  << "number of G-vectors: " << fft__->num_gvec();
                  TERMINATE(s);
            }
        }
    }
}

void Symmetry::symmetrize_function(double_complex* f_pw__,
                                   FFT3D<CPU>* fft__,
                                   splindex<block>& spl_num_gvec__,
                                   Communicator& comm__)
{
    Timer t("sirius::Symmetry::symmetrize_function");
    mdarray<double_complex, 1> sym_f_pw(fft__->num_gvec());
    sym_f_pw.zero();
    double* ptr = (double*)&sym_f_pw(0);

    #pragma omp parallel for
    for (int isym = 0; isym < num_sym_op(); isym++)
    {
        /* full symmetry operation is {R|t} */
        auto R = rot_mtrx(isym);
        auto t = fractional_translation(isym);

        for (int igloc = 0; igloc < (int)spl_num_gvec__.local_size(); igloc++)
        {
            int ig = (int)spl_num_gvec__[igloc];
            /* apply symmetry operation to the G-vector;
             * remember that we move R from acting on x to acting on G: G(Rx) = (GR)x;
             * GR is a vector-matrix multiplication [G][.....]
             *                                         [..R..]
             *                                         [.....]
             * which can also be written as matrix^{T}-vector operation
             */
            vector3d<int> gv_rot = transpose(R) * fft__->gvec(ig);

            /* index of a rotated G-vector */
            int ig_rot = fft__->gvec_index(gv_rot);

            assert(ig_rot >= 0 && ig_rot < fft__->num_gvec());

            double_complex z = f_pw__[ig] * std::exp(double_complex(0, twopi * (fft__->gvec(ig) * t)));
            
            #pragma omp atomic update
            ptr[2 * ig_rot] += real(z);

            #pragma omp atomic update
            ptr[2 * ig_rot + 1] += imag(z);
        }
    }
    comm__.allreduce(&sym_f_pw(0), fft__->num_gvec());

    for (int ig = 0; ig < fft__->num_gvec(); ig++) f_pw__[ig] = sym_f_pw(ig) / double(num_sym_op());
}

void Symmetry::symmetrize_vector_z_component(double_complex* f_pw__,
                                   FFT3D<CPU>* fft__,
                                   Communicator& comm__)
{
    Timer t("sirius::Symmetry::symmetrize_vector_z_component");
    
    splindex<block> spl_gvec(fft__->num_gvec(), comm__.size(), comm__.rank());
    mdarray<double_complex, 1> sym_f_pw(fft__->num_gvec());
    sym_f_pw.zero();

    double* ptr = (double*)&sym_f_pw(0);

    #pragma omp parallel for
    for (int i = 0; i < (int)mag_sym_.size(); i++)
    {
        int isym = mag_sym_[i].first;
        int jsym = mag_sym_[i].second;

        /* full symmetry operation is {R|t} */
        auto R = rot_mtrx(isym);
        auto t = fractional_translation(isym);

        /* take proper part of rotation matrix */
        auto Rspin = rot_mtrx_cart(jsym);
        Rspin = Rspin * Rspin.det();

        for (int igloc = 0; igloc < (int)spl_gvec.local_size(); igloc++)
        {
            int ig = (int)spl_gvec[igloc];

            vector3d<int> gv_rot = transpose(R) * fft__->gvec(ig);

            /* index of a rotated G-vector */
            int ig_rot = fft__->gvec_index(gv_rot);

            assert(ig_rot >= 0 && ig_rot < fft__->num_gvec());

            double_complex z = f_pw__[ig] * std::exp(double_complex(0, twopi * (fft__->gvec(ig) * t))) * Rspin(2, 2);
            
            #pragma omp atomic update
            ptr[2 * ig_rot] += real(z);

            #pragma omp atomic update
            ptr[2 * ig_rot + 1] += imag(z);
        }
    }
    comm__.allreduce(&sym_f_pw(0), fft__->num_gvec());

    for (int ig = 0; ig < fft__->num_gvec(); ig++) f_pw__[ig] = sym_f_pw(ig) / double(mag_sym_.size());

}

};

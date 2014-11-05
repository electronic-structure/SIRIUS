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

Symmetry::Symmetry(double lattice_vectors__[3][3], SpglibDataset* spg_dataset__) 
    : spg_dataset_(spg_dataset__)
{
    assert(spg_dataset__ != NULL);

    lattice_vectors_ = transpose(matrix3d<double>(lattice_vectors__));

    inverse_lattice_vectors_ = inverse(lattice_vectors_);
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

    if (type_wrapper<double>::abs(rm(2, 2) - 1.0) < eps) // cos(beta) == 1, beta = 0
    {
        angles[0] = Utils::phi_by_sin_cos(rm(1, 0), rm(0, 0));
    }
    else if (type_wrapper<double>::abs(rm(2, 2) + 1.0) < eps) // cos(beta) == -1, beta = Pi
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
            if (type_wrapper<double>::abs(rm(i, j) - rm1(i, j)) > eps)
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
            
}


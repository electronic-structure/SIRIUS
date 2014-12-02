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

/** \file symmetry.h
 *   
 *  \brief Contains definition and partial implementation of sirius::Symmetry class.
 */

#ifndef __SYMMETRY_H__
#define __SYMMETRY_H__

extern "C" {
#include <spglib.h>
}

#include "vector3d.h"
#include "matrix3d.h"
#include "constants.h"
#include "utils.h"
#include "fft3d.h"

namespace sirius {

class Symmetry
{
    private:
        
        matrix3d<double> lattice_vectors_;

        matrix3d<double> inverse_lattice_vectors_;

        SpglibDataset* spg_dataset_;

    public:

        Symmetry()
        {
        }

        Symmetry(matrix3d<double> lattice_vectors__, SpglibDataset* spg_dataset__);
        
        inline int num_sym_op()
        {
            return spg_dataset_->n_operations;
        }

        int proper_rotation(int isym);

        /// Rotation matrix in Cartesian coordinates.
        matrix3d<double> rot_mtrx_cart(int isym__);

        /// Rotation matrix in fractional coordinates.
        matrix3d<int> rot_mtrx(int isym__);
        
        /// Generate rotation matrix from three Euler angles
        /** Euler angles \f$ \alpha, \beta, \gamma \f$ define the general rotation as three consecutive rotations:
         *      - about \f$ \hat e_z \f$ through the angle \f$ \gamma \f$ (\f$ 0 \le \gamma < 2\pi \f$)
         *      - about \f$ \hat e_y \f$ through the angle \f$ \beta \f$ (\f$ 0 \le \beta \le \pi \f$) 
         *      - about \f$ \hat e_z \f$ through the angle \f$ \alpha \f$ (\f$ 0 \le \gamma < 2\pi \f$)
         *  
         *  The total rotation matrix is defined as a product of three rotation matrices:
         *  \f[
         *      R(\alpha, \beta, \gamma) = 
         *          \left( \begin{array}{ccc} \cos(\alpha) & -\sin(\alpha) & 0 \\
         *                                    \sin(\alpha) & \cos(\alpha) & 0 \\
         *                                    0 & 0 & 1 \end{array} \right) 
         *          \left( \begin{array}{ccc} \cos(\beta) & 0 & \sin(\beta) \\
         *                                    0 & 1 & 0 \\
         *                                    -\sin(\beta) & 0 & \cos(\beta) \end{array} \right)  
         *          \left( \begin{array}{ccc} \cos(\gamma) & -\sin(\gamma) & 0 \\
         *                                    \sin(\gamma) & \cos(\gamma) & 0 \\
         *                                    0 & 0 & 1 \end{array} \right) = 
         *      \left( \begin{array}{ccc} \cos(\alpha) \cos(\beta) \cos(\gamma) - \sin(\alpha) \sin(\gamma) & 
         *                                -\sin(\alpha) \cos(\gamma) - \cos(\alpha) \cos(\beta) \sin(\gamma) & 
         *                                \cos(\alpha) \sin(\beta) \\
         *                                \sin(\alpha) \cos(\beta) \cos(\gamma) + \cos(\alpha) \sin(\gamma) & 
         *                                \cos(\alpha) \cos(\gamma) - \sin(\alpha) \cos(\beta) \sin(\gamma) & 
         *                                \sin(\alpha) \sin(\beta) \\
         *                                -\sin(\beta) \cos(\gamma) & 
         *                                \sin(\beta) \sin(\gamma) & 
         *                                \cos(\beta) \end{array} \right)
         *  \f]
         */
        matrix3d<double> rot_mtrx(vector3d<double> euler_angles);
        
        /// Compute Euler angles corresponding to the proper rotation part of the given symmetry.
        /** 

        */
        vector3d<double> euler_angles(int isym__);

        vector3d<double> fractional_translation(int isym__)
        {
            vector3d<double> t;
            for (int x = 0; x < 3; x++) t[x] =  spg_dataset_->translations[isym__][x];
            return t;
        }

        void check_gvec_symmetry(FFT3D<CPU>* fft__)
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
                            s << "rotated G-vector is outside grid limits" << std::endl
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

        void symmetrize_function(double_complex* f_pw__, FFT3D<CPU>* fft__)
        {
            mdarray<double_complex, 1> sym_f_pw(fft__->num_gvec());
            sym_f_pw.zero();

            for (int isym = 0; isym < num_sym_op(); isym++)
            {
                auto sm = rot_mtrx(isym);
                auto sv = fractional_translation(isym);

                for (int ig = 0; ig < fft__->num_gvec(); ig++)
                {
                    /* apply symmetry operation to the G-vector */
                    vector3d<int> gv_rot = transpose(sm) * fft__->gvec(ig);

                    /* index of a rotated G-vector */
                    int ig_rot = fft__->gvec_index(gv_rot);

                    assert(ig_rot >= 0 && ig_rot < fft__->num_gvec());

                    sym_f_pw(ig_rot) += f_pw__[ig] * std::exp(double_complex(0, -twopi * (gv_rot * sv)));
                }
            }

            for (int ig = 0; ig < fft__->num_gvec(); ig++) f_pw__[ig] = sym_f_pw(ig) / double(num_sym_op());
        }
};

}

/** \page sym Symmetry
    \section section1 Definition of symmetry operation

    SIRIUS uses Spglib to find the spacial symmetry operations. Spglib defines symmetry operation in fractional 
    coordinates:
    \f[
        {\bf x'} = \{ {\bf R} | {\bf t} \} {\bf x} \equiv {\bf R}{\bf x} + {\bf t}
    \f]
    where \b R is the proper or improper rotation matrix with elements equal to -1,0,1 and determinant of 1 
    (pure rotation) or -1 (rotoreflection) and \b t is the fractional translation, associated with the symmetry 
    operation. The inverse of the symmetry operation is:
    \f[
        {\bf x} = \{ {\bf R} | {\bf t} \}^{-1} {\bf x'} = {\bf R}^{-1} ({\bf x'} - {\bf t}) = 
            {\bf R}^{-1} {\bf x'} - {\bf R}^{-1} {\bf t}
    \f]

    We will always use an \a active transformation (transformation of vectors or functions) and never a passive
    transformation (transformation of coordinate system). However one should remember definition of the function
    transformation:
    \f[
        \hat {\bf P} f({\bf r}) \equiv f(\hat {\bf P}^{-1} {\bf r})
    \f]

    It is straightforward to get the rotation matrix in Cartesian coordinates. We know how the vector in Cartesian 
    coordinates is obtained from the vector in fractional coordinates:
    \f[
        {\bf v} = {\bf L} {\bf x}
    \f]
    where \b L is the 3x3 matrix which clomuns are three lattice vectors. The backward transformation is simply
    \f[
        {\bf x} = {\bf L}^{-1} {\bf v}
    \f]
    Now we write rotation operation in fractional coordinates and apply the backward transformation to Cartesian 
    coordinates:
    \f[
        {\bf x'} = {\bf R}{\bf x} \rightarrow {\bf L}^{-1} {\bf v'} = {\bf R} {\bf L}^{-1} {\bf v}
    \f]
    from which we derive the rotation operation in Cartesian coordinates:
    \f[
        {\bf v'} = {\bf L} {\bf R} {\bf L}^{-1} {\bf v}
    \f]
*/

#endif // __SYMMETRY_H__

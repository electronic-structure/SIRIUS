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

struct space_group_symmetry_descriptor
{
    /// Rotational part of symmetry operation (fractional coordinates).
    matrix3d<int> R;

    /// Fractional translation.
    vector3d<double> t;
    
    /// Proper (+1) or improper (-1) rotation.
    int proper;

    /// Proper rotation matrix in Cartesian coordinates.
    matrix3d<double> rotation;

    vector3d<double> euler_angles;
};

struct magnetic_group_symmetry_descriptor
{
    space_group_symmetry_descriptor spg_op;

    int isym;

    matrix3d<double> spin_rotation;
};

class Symmetry
{
    private:
        
        matrix3d<double> lattice_vectors_;

        matrix3d<double> inverse_lattice_vectors_;

        int num_atoms_;

        mdarray<double, 2> positions_;

        std::vector<int> types_;

        double tolerance_;

        SpglibDataset* spg_dataset_;

        //std::vector< std::pair<int, int> > mag_sym_;

        mdarray<int, 2> sym_table_;
        
        std::vector<space_group_symmetry_descriptor> space_group_symmetry_;

        std::vector<magnetic_group_symmetry_descriptor> magnetic_group_symmetry_;

        /// Compute Euler angles corresponding to the proper rotation part of the given symmetry.
        /** 

        */
        //vector3d<double> euler_angles(int isym__);

        vector3d<double> euler_angles(matrix3d<double> const& rot__) const;

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
        matrix3d<double> rot_mtrx_cart(vector3d<double> euler_angles__) const;

    public:

        Symmetry(matrix3d<double>& lattice_vectors__,
                 int num_atoms__,
                 mdarray<double, 2>& positions__,
                 mdarray<double, 2>& spins__,
                 std::vector<int>& types__,
                 double tolerance__);

        ~Symmetry();

        //inline int num_sym_op()
        //{
        //    return spg_dataset_->n_operations;
        //}

        inline int atom_symmetry_class(int ia__)
        {
            return spg_dataset_->equivalent_atoms[ia__];
        }

        inline int spacegroup_number()
        {
            return spg_dataset_->spacegroup_number;
        }

        inline std::string international_symbol()
        {
            return spg_dataset_->international_symbol;
        }

        inline std::string hall_symbol()
        {
            return spg_dataset_->hall_symbol;
        }

        matrix3d<double> transformation_matrix() const
        {
           return matrix3d<double>(spg_dataset_->transformation_matrix);
        }

        vector3d<double> origin_shift() const
        {
            return vector3d<double>(spg_dataset_->origin_shift);
        }

        inline int num_spg_sym() const
        {
            return (int)space_group_symmetry_.size();
        }

        inline space_group_symmetry_descriptor const& space_group_symmetry(int isym__) const
        {
            assert(isym__ >= 0 && isym__ < num_spg_sym());
            return space_group_symmetry_[isym__];
        }
        inline int num_mag_sym() const
        {
            return (int)magnetic_group_symmetry_.size();
        }

        inline magnetic_group_symmetry_descriptor const& magnetic_group_symmetry(int isym__) const
        {
            assert(isym__ >= 0 && isym__ < num_mag_sym());
            return magnetic_group_symmetry_[isym__];
        }

        //int proper_rotation(int isym);

        /// Rotation matrix in Cartesian coordinates.
        //matrix3d<double> rot_mtrx_cart(int isym__);

        /// Rotation matrix in fractional coordinates.
        //matrix3d<int> rot_mtrx(int isym__);
        
        
        //vector3d<double> fractional_translation(int isym__)
        //{
        //    vector3d<double> t;
        //    for (int x = 0; x < 3; x++) t[x] =  spg_dataset_->translations[isym__][x];
        //    return t;
        //}

        void check_gvec_symmetry(FFT3D<CPU>* fft__) const;

        /// Symmetrize scalar function.
        /** The following operation is performed:
         *  \f[
         *    f({\bf x}) = \frac{1}{N_{sym}} \sum_{{\bf \hat P}} f({\bf \hat P x})
         *  \f]
         *  For the function expanded in plane-waves we have:
         *  \f[
         *    f({\bf x}) = \frac{1}{N_{sym}} \sum_{{\bf \hat P}} \sum_{\bf G} e^{i{\bf G \hat P x}} f({\bf G}) 
         *               = \frac{1}{N_{sym}} \sum_{{\bf \hat P}} \sum_{\bf G} e^{i{\bf G (Rx + t)}} f({\bf G})
         *               = \frac{1}{N_{sym}} \sum_{{\bf \hat P}} \sum_{\bf G} e^{i{\bf G t}} e^{i{\bf G Rx}} f({\bf G})
         *  \f]
         *  Now we do a mapping \f$ {\bf GR} \rightarrow \tilde {\bf G} \f$ and find expansion coefficients of the
         *  symmetry transformed function:
         *  \f[
         *    f(\tilde{\bf G}) = e^{i{\bf G t}} f({\bf G})
         *  \f]
         */
        void symmetrize_function(double_complex* f_pw__,
                                 FFT3D<CPU>* fft__,
                                 Communicator const& comm__) const;
        
        void symmetrize_function(mdarray<double, 3>& frlm__,
                                 Communicator const& comm__) const;
        
        void symmetrize_vector_z_component(double_complex* f_pw__,
                                           FFT3D<CPU>* fft__,
                                           Communicator const& comm__) const;

        void symmetrize_vector_z_component(mdarray<double, 3>& frlm__,
                                           Communicator const& comm__) const;

        int get_irreducible_reciprocal_mesh(vector3d<int> k_mesh__,
                                            vector3d<int> is_shift__,
                                            mdarray<double, 2>& kp__,
                                            std::vector<double>& wk__) const;
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

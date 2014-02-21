// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __SYMMETRY_H__
#define __SYMMETRY_H__

/** \file symmetry.h
    
    \brief Contains definition and partial implementation of sirius::Symmetry class.
*/

extern "C" {
#include <spglib.h>
}

#include "vector3d.h"
#include "matrix3d.h"
#include "constants.h"
#include "utils.h"

namespace sirius {

class Symmetry
{
    private:
        
        matrix3d<double> lattice_vectors_;

        matrix3d<double> inverse_lattice_vectors_;

        SpglibDataset* spg_dataset_;

    public:

        Symmetry(double lattice_vectors__[3][3], SpglibDataset* spg_dataset__);
        
        inline int num_sym_op()
        {
            return spg_dataset_->n_operations;
        }

        int proper_rotation(int isym);

        matrix3d<double> rot_mtrx(int isym);

        matrix3d<double> rot_mtrx(vector3d<double> euler_angles);

        vector3d<double> euler_angles(int isym);
};

}

/** \page sym Symmetry
    \section section1 Definition of symmetry operation

    SIRIUS uses Spglib to find the spacial symmetry operations. In Spglib symmetry operation is defined in lattice 
    coordinates:
    \f[
        {\bf x'} = {\bf R}{\bf x} + {\bf t}
    \f]
    where \b R is the proper or improper rotation matrix with elements equal to -1,0,1 and determinant of 1 
    (pure rotation) or -1 (rotoreflection) and \b t is the partial translation in fractional coordinates, associated
    with the symmetry operation.

*/

#endif // __SYMMETRY_H__

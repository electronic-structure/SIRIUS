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

/** \file descriptors.h
 *
 *  \brief Descriptors for various data structures
 */

#ifndef __DESCRIPTORS_H__
#define __DESCRIPTORS_H__

#include "mdarray.h"
#include "vector3d.h"
#include "utils.h"

/// Describes single atomic level.
struct atomic_level_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;
    
    /// Quantum number k.
    int k;
    
    /// Level occupancy.
    double occupancy;

    /// True if this is a core level.
    bool core;
};

/// Describes radial solution.
struct radial_solution_descriptor
{
    /// Principal quantum number.
    int n;
    
    /// Angular momentum quantum number.
    int l;
    
    /// Order of energy derivative.
    int dme;
    
    /// Energy of the solution.
    double enu;
    
    /// Automatically determine energy.
    int auto_enu;
};

/// Set of radial solution descriptors, used to construct augmented waves or local orbitals.
typedef std::vector<radial_solution_descriptor> radial_solution_descriptor_set;

/// descriptor of a local orbital radial function
struct local_orbital_descriptor
{
    int l;

    radial_solution_descriptor_set rsd_set;

    int p1;
    int p2;
};

struct uspp_descriptor
{
    /// Radial mesh.
    std::vector<double> r;

    /// Local part of potential.
    std::vector<double> vloc;

    /// Maximum angular momentum for |beta> projectors.
    int lmax_beta_;

    /// Number of radial functions for |beta> projectors.
    int num_beta_radial_functions;

    /// Orbital quantum numbers of each beta radial function.
    std::vector<int> beta_l;

    /// Number of radial grid points for each beta radial function.
    std::vector<int> num_beta_radial_points;

    /// Radial functions of beta-projectors.
    mdarray<double, 2> beta_radial_functions;

    /// Radial functions of Q-operator.
    mdarray<double, 3> q_radial_functions_l;

    bool augmentation_{false};

    std::vector<double> core_charge_density;

    std::vector<double> total_charge_density;

    mdarray<double, 2> d_mtrx_ion;

    mdarray<double, 2> wf_pseudo_;

    std::vector<int> l_wf_pseudo_;

    /// Atomic wave-functions used to setup the initial subspace.
    /** This are the chi wave-function in the USPP file. Pairs of [l, chi_l(r)] are stored. */
    std::vector< std::pair<int, std::vector<double> > > atomic_pseudo_wfs_;
    
    /// occupation of starting wave functions
    std::vector< double > atomic_pseudo_wfs_occ_;

    bool is_initialized{false};
};

struct nearest_neighbour_descriptor
{
    /// id of neighbour atom
    int atom_id;

    /// translation along each lattice vector
    vector3d<int> translation;

    /// distance from the central atom
    double distance;
};

struct radial_function_index_descriptor
{
    int l;

    int order;

    int idxlo;

    radial_function_index_descriptor(int l, int order, int idxlo = -1) 
        : l(l), 
          order(order), 
          idxlo(idxlo)
    {
        assert(l >= 0);
        assert(order >= 0);
    }
};

struct basis_function_index_descriptor
{
    int l;

    int m;

    int lm;

    int order;

    int idxlo;

    int idxrf;
    
    basis_function_index_descriptor(int l, int m, int order, int idxlo, int idxrf) 
        : l(l), 
          m(m), 
          order(order), 
          idxlo(idxlo), 
          idxrf(idxrf) 
    {
        assert(l >= 0);
        assert(m >= -l && m <= l);
        assert(order >= 0);
        assert(idxrf >= 0);

        lm = Utils::lm_by_l_m(l, m);
    }
};

struct unit_cell_parameters_descriptor
{
    double a;
    double b;
    double c;
    double alpha;
    double beta;
    double gamma;
};

/// Descriptor of the G+k or local-orbital basis function.
/** This data structure describes one of the following basis sets:
 *      - LAPW+lo basis, which consists of G+k labeled augmented plane-waves and of local orbitals
 *      - PW+lo basis, which consists of G+k plane-waves and of local orbitals
 *      - pure G+k plane-wave basis */
struct gklo_basis_descriptor
{
    vector3d<int> gvec;

    /// G+k vector in fractional coordinates.
    vector3d<double> gkvec;

    /// Global index of the G vector for the corresponding G+k vector.
    int ig;

    /// Index of atom if this is a local orbital descriptor.
    uint16_t ia;

    /// Index of orbital quantum number \f$ \ell \f$.
    uint8_t l;

    /// Combined lm index.
    uint16_t lm;

    /// Order of the local orbital radial function for the given orbital quantum number \f$ \ell \f$.
    /** All radial functions for the given orbital quantum number \f$ \ell \f$ are ordered in the following way: 
     *  augmented radial functions come first followed by the local orbital radial function. */
    uint8_t order;

    /// Index of the local orbital radial function.
    uint8_t idxrf;
};

struct mt_basis_descriptor
{
    int ia;
    int xi;
};

struct block_data_descriptor
{
    int num_ranks{-1};
    std::vector<int> counts;
    std::vector<int> offsets;

    block_data_descriptor()
    {
    }

    block_data_descriptor(int num_ranks__) : num_ranks(num_ranks__)
    {
        counts  = std::vector<int>(num_ranks, 0);
        offsets = std::vector<int>(num_ranks, 0);
    }

    void calc_offsets()
    {
        for (int i = 1; i < num_ranks; i++) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
    }
};

struct z_column_descriptor
{
    int x;
    int y;

    /// Z-coordinates of the column.
    std::vector<int> z;

    z_column_descriptor()
    {
    }

    z_column_descriptor(int x__, int y__, std::vector<int> z__) : x(x__), y(y__), z(z__)
    {
    }
};

#endif

// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

#include "mdarray.hpp"
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

/// Descriptor of a local orbital radial function.
struct local_orbital_descriptor
{
    /// Orbital quantum number \f$ \ell \f$.
    int l;

    /// Total angular momentum
    double j;

    /// Set of radial solution descriptors.
    /** Local orbital is constructed from at least two radial functions in order to make it zero at the 
     *  muffin-tin sphere boundary. */
    radial_solution_descriptor_set rsd_set;
};

/// Descriptor of the pseudopotential.
struct pseudopotential_descriptor
{
    /// The pseudo potential includes spin orbit coupling
    bool SpinOrbit_Coupling{false};
  
    /// True if the pseudopotential is soft and charge augmentation is required.
    bool augment{false};
    
    /// True if the pseudopotential is used for PAW.
    bool is_paw{false};

    /// Local part of potential.
    std::vector<double> vloc;

    /// Maximum angular momentum for |beta> projectors.
    int lmax_beta_;

    /// Number of radial functions for |beta> projectors.
    int num_beta_radial_functions;

    /// Orbital quantum numbers of each beta radial function.
    std::vector<int> beta_l;

    /// Total orbital quantum numbers of each beta radial function.
    std::vector<int> beta_j;

    /// Number of radial grid points for each beta radial function.
    std::vector<int> num_beta_radial_points;

    /// Radial functions of beta-projectors.
    mdarray<double, 2> beta_radial_functions;

    /// Radial functions of Q-operator.
    mdarray<double, 3> q_radial_functions_l;

    std::vector<double> core_charge_density;

    std::vector<double> total_charge_density;

    mdarray<double, 2> d_mtrx_ion;

    /// Atomic wave-functions used to setup the initial subspace.
    /** This are the chi wave-function in the USPP file. Pairs of [l, chi_l(r)] are stored. */
    std::vector<std::pair<int, std::vector<double>>> atomic_pseudo_wfs_;
    
    /// Occupation of starting wave functions
    //std::vector<double> atomic_pseudo_wfs_occ_;

    /// All electron basis wave functions, have the same dimensionality as uspp.beta_radial_functions.
    mdarray<double, 2> all_elec_wfc;

    /// pseudo basis wave functions, have the same dimensionality as uspp.beta_radial_functions
    mdarray<double, 2> pseudo_wfc;

    /// Core energy of PAW.
    double core_energy; // TODO: proper desciption comment

    /// Occubations of atomic states. 
    /** Length of vector is the same as the number of beta projectors and all_elec_wfc and pseudo_wfc */
    std::vector<double> occupations;

    /// density of core electron contribution to all electron charge density
    std::vector<double> all_elec_core_charge;

    /// electrostatic potential of all electron core charge
    std::vector<double> all_elec_loc_potential;

    int cutoff_radius_index;
};

/// Descriptor of an atom in a list of nearest neigbours for each atom.
/** See sirius::Unit_cell::find_nearest_neighbours() for the details of usage. */
struct nearest_neighbour_descriptor
{
    /// Index of the neighbouring atom.
    int atom_id;

    /// Translation in fractional coordinates.
    geometry3d::vector3d<int> translation;

    /// Distance from the central atom.
    double distance;
};

/// Descriptor for the atomic radial functions.
/** The radial functions \f$ f_{\ell \nu}(r) \f$ are labeled by two indices: orbital quantum number \f$ \ell \f$ and
 *  an order \f$ \nu \f$ for a given $\f \ell \f$. 
 */
struct radial_function_index_descriptor
{
    /// Orbital quantum number \f$ \ell \f$.
    int l;

  /// Total angular momentum
  double j;
    /// Order of a function for a given \f$ \ell \f$.
    int order;

    /// If this is a local orbital radial function, idxlo is it's index in the list of local orbital descriptors.
    int idxlo;
    
    /// Constructor.
    radial_function_index_descriptor(int l, int order, int idxlo = -1) 
        : l(l)
        , order(order)
        , idxlo(idxlo)
    {
        assert(l >= 0);
        assert(order >= 0);
    }

    radial_function_index_descriptor(int l, double j, int order, int idxlo = -1) 
        : l(l),
          j(j),
          order(order), 
          idxlo(idxlo)
    {
        assert(l >= 0);
        assert(order >= 0);
    }
};

struct basis_function_index_descriptor
{
    /// angular momentum
    int l;
    /// projection of the angular momentum
    int m;
    /// composite index
    int lm;
    /// total angular momemtum
    double j;
    /// order of the radial function for a given l (j)
    int order;
    /// indice of local orbital
    int idxlo;
    /// index of the radial function or beta projector in the case of
    /// pseudo potential
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

    basis_function_index_descriptor(int l, int m, double j, int order, int idxlo, int idxrf) 
         : l(l), 
           m(m),
           j(j),
           order(order), 
           idxlo(idxlo), 
           idxrf(idxrf) 
    {
         assert(l >= 0);
         assert(m >= -l && m <= l);
         assert(order >= 0);
         assert(idxrf >= 0);
         assert(std:fabs(std::fabs(j)-0.5)< 1e-8);
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

/// Descriptor of the local-orbital part of the LAPW+lo basis. 
struct lo_basis_descriptor
{
    /// Index of atom.
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

#endif

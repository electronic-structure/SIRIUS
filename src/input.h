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

/** \file input.h
 *   
 *  \brief Contains declarations and implementations of input parameters structures.
 */

#ifndef __INPUT_H__
#define __INPUT_H__

#include <omp.h>
#include "json_tree.h"

namespace sirius {

/// Parse unit cell input section.
/** The following part of the input file is parsed:
 *  \code{.json}
 *      "unit_cell" : {
 *          "lattice_vectors" : [
 *              [a1_x, a1_y, a1_z],
 *              [a2_x, a2_y, a2_z],
 *              [a3_x, a3_y, a3_z]
 *          ],
 *
 *          "lattice_vectors_scale" : scale,
 *
 *          "atom_types" : [label_A, label_B, ...],
 *
 *          "atom_files" : {
 *              label_A : file_A, 
 *              label_B : file_B,
 *              ...
 *          },
 *
 *          "atoms" : {
 *              label_A: [
 *                  coordinates_A_1, 
 *                  coordinates_A_2,
 *                  ...
 *              ],
 *              label_B : [
 *                  coordinates_B_1,
 *                  coordinates_B_2,
 *                  ...
 *              ]
 *          }
 *      }
 *  \endcode
 */
struct Unit_cell_input_section
{
    double lattice_vectors_[3][3];

    std::vector<std::string> labels_;
    std::map<std::string, std::string> atom_files_;
    std::vector< std::vector< std::vector<double> > > coordinates_;

    bool exist_{false};

    void read(JSON_tree const& parser)
    {
        if (parser.exist("unit_cell"))
        {
            exist_ = true;

            auto section = parser["unit_cell"];
            std::vector<double> a0, a1, a2;
            section["lattice_vectors"][0] >> a0;
            section["lattice_vectors"][1] >> a1;
            section["lattice_vectors"][2] >> a2;

            if (a0.size() != 3 || a1.size() != 3 || a2.size() != 3)
                TERMINATE("wrong lattice vectors");

            double scale = section["lattice_vectors_scale"].get(1.0);

            for (int x = 0; x < 3; x++)
            {
                lattice_vectors_[0][x] = a0[x] * scale;
                lattice_vectors_[1][x] = a1[x] * scale;
                lattice_vectors_[2][x] = a2[x] * scale;
            }

            labels_.clear();
            coordinates_.clear();
            
            for (int iat = 0; iat < (int)section["atom_types"].size(); iat++)
            {
                std::string label;
                section["atom_types"][iat] >> label;
                for (int i = 0; i < (int)labels_.size(); i++)
                {
                    if (labels_[i] == label) 
                        TERMINATE("atom type with such label is already in list");
                }
                labels_.push_back(label);
            }
            
            if (section.exist("atom_files"))
            {
                for (int iat = 0; iat < (int)labels_.size(); iat++)
                    atom_files_[labels_[iat]] = section["atom_files"][labels_[iat]].get(std::string(""));
            }
            
            for (int iat = 0; iat < (int)labels_.size(); iat++)
            {
                coordinates_.push_back(std::vector< std::vector<double> >());
                for (int ia = 0; ia < section["atoms"][labels_[iat]].size(); ia++)
                {
                    std::vector<double> v;
                    section["atoms"][labels_[iat]][ia] >> v;

                    if (!(v.size() == 3 || v.size() == 6)) TERMINATE("wrong coordinates size");
                    if (v.size() == 3) v.resize(6, 0.0);

                    coordinates_[iat].push_back(v);
                }
            }
        }
    }
};

struct Mixer_input_section
{
    double beta_{0.9};
    double beta0_{0.15};
    double linear_mix_rms_tol_{1e6};
    std::string type_{"broyden2"};
    int max_history_{8};
    bool exist_{false};

    void read(JSON_tree const& parser)
    {
        if (parser.exist("mixer"))
        {
            exist_ = true;
            auto section = parser["mixer"];
            beta_               = section["beta"].get(beta_);
            beta0_              = section["beta0"].get(beta0_);
            linear_mix_rms_tol_ = section["linear_mix_rms_tol"].get(linear_mix_rms_tol_);
            max_history_        = section["max_history"].get(max_history_);
            type_               = section["type"].get(type_);
        }
    }
};

/** \todo real-space projectors are not part of iterative solver */
struct Iterative_solver_input_section
{
    std::string type_{"davidson"};
    int num_steps_{20};
    int subspace_size_{4};
    double energy_tolerance_{1e-6};
    double residual_tolerance_{1e-6};
    int converge_by_energy_{1}; // TODO: rename, this is meaningless
    int converge_occupied_{1};
    int min_num_res_{0};
    int real_space_prj_{0}; // TODO: move it from here to parameters
    double R_mask_scale_{1.5};
    double mask_alpha_{3};

    void read(JSON_tree const& parser)
    {
        type_               = parser["iterative_solver"]["type"].get(type_);
        num_steps_          = parser["iterative_solver"]["num_steps"].get(num_steps_);
        subspace_size_      = parser["iterative_solver"]["subspace_size"].get(subspace_size_);
        energy_tolerance_   = parser["iterative_solver"]["energy_tolerance"].get(energy_tolerance_);
        residual_tolerance_ = parser["iterative_solver"]["residual_tolerance"].get(residual_tolerance_);
        converge_by_energy_ = parser["iterative_solver"]["converge_by_energy"].get(converge_by_energy_);
        converge_occupied_  = parser["iterative_solver"]["converge_occupied"].get(converge_occupied_);
        min_num_res_        = parser["iterative_solver"]["min_num_res"].get(min_num_res_);
        real_space_prj_     = parser["iterative_solver"]["real_space_prj"].get(real_space_prj_);
        R_mask_scale_       = parser["iterative_solver"]["R_mask_scale"].get(R_mask_scale_);
        mask_alpha_         = parser["iterative_solver"]["mask_alpha"].get(mask_alpha_);
    }
};

/// Parse control input section.
/** The following part of the input file is parsed:
 *  \code{.json}
 *    "control" : {
 *      "mpi_grid_dims" : (1- 2- or 3-dimensional vector<int>) MPI grid layout
 *      "cyclic_block_size" : (int) PBLAS / ScaLAPACK block size
 *      "reduce_gvec" : (int) use reduced G-vector set (reduce_gvec = 1) or full set (reduce_gvec = 0)
 *      "std_evp_solver_type" : (string) type of eigen-solver for the standard eigen-problem
 *      "gen_evp_solver_type" : (string) type of eigen-solver for the generalized eigen-problem
 *      "electronic_structure_method" : (string) electronic structure method
 *      "processing_unit" : (string) primary processing unit
 *      "fft_mode" : (string) serial or parallel FFT
 *    }
 *  \endcode
 */
struct Control_input_section
{
    std::vector<int> mpi_grid_dims_;
    int cyclic_block_size_{32};
    bool reduce_gvec_{true};
    std::string std_evp_solver_name_{"lapack"};
    std::string gen_evp_solver_name_{"lapack"};
    std::string fft_mode_{"serial"};
    std::string processing_unit_{"cpu"};
    double rmt_max_{2.2};
    double spglib_tolerance_{1e-4};

    void read(JSON_tree const& parser)
    {
        mpi_grid_dims_       = parser["control"]["mpi_grid_dims"].get(mpi_grid_dims_); 
        cyclic_block_size_   = parser["control"]["cyclic_block_size"].get(cyclic_block_size_);
        std_evp_solver_name_ = parser["control"]["std_evp_solver_type"].get(std_evp_solver_name_);
        gen_evp_solver_name_ = parser["control"]["gen_evp_solver_type"].get(gen_evp_solver_name_);

        processing_unit_ = parser["control"]["processing_unit"].get(processing_unit_);
        std::transform(processing_unit_.begin(), processing_unit_.end(), processing_unit_.begin(), ::tolower);

        fft_mode_ = parser["control"]["fft_mode"].get(fft_mode_);
        reduce_gvec_ = parser["control"]["reduce_gvec"].get<int>(reduce_gvec_);

        rmt_max_ = parser["control"]["rmt_max"].get(rmt_max_);
        spglib_tolerance_ = parser["control"]["spglib_tolerance"].get(spglib_tolerance_);
    }
};

struct Parameters_input_section
{
    std::string esm_{"none"};
    std::vector<std::string> xc_functionals_;
    std::string core_relativity_{"dirac"};
    std::string valence_relativity_{"zora"};
    int num_fv_states_{-1};
    double smearing_width_{0.01}; // in Ha
    double pw_cutoff_{20.0}; // in a.u.^-1
    double aw_cutoff_{7.0}; // this is R_{MT} * |G+k|_{max}
    double gk_cutoff_{6.0}; // in a.u.^-1
    int lmax_apw_{10};
    int lmax_rho_{10};
    int lmax_pot_{10};
    int num_mag_dims_{0};
    int auto_rmt_{1};
    int use_symmetry_{1};
    int gamma_point_{0};

    void read(JSON_tree const& parser)
    {
        esm_ = parser["parameters"]["electronic_structure_method"].get(esm_);
        std::transform(esm_.begin(), esm_.end(), esm_.begin(), ::tolower);

        /* read list of XC functionals */
        if (parser["parameters"].exist("xc_functionals")) {
            xc_functionals_.clear();
            for (int i = 0; i < parser["parameters"]["xc_functionals"].size(); i++) {
                std::string s;
                parser["parameters"]["xc_functionals"][i] >> s;
                xc_functionals_.push_back(s);
            }
        }

        core_relativity_ = parser["parameters"]["core_relativity"].get(core_relativity_);
        std::transform(core_relativity_.begin(), core_relativity_.end(), core_relativity_.begin(), ::tolower);

        valence_relativity_ = parser["parameters"]["valence_relativity"].get(valence_relativity_);
        std::transform(valence_relativity_.begin(), valence_relativity_.end(), valence_relativity_.begin(), ::tolower);

        num_fv_states_  = parser["parameters"]["num_fv_states"].get(num_fv_states_);
        smearing_width_ = parser["parameters"]["smearing_width"].get(smearing_width_);
        pw_cutoff_      = parser["parameters"]["pw_cutoff"].get(pw_cutoff_);
        aw_cutoff_      = parser["parameters"]["aw_cutoff"].get(aw_cutoff_);
        gk_cutoff_      = parser["parameters"]["gk_cutoff"].get(gk_cutoff_);
        lmax_apw_       = parser["parameters"]["lmax_apw"].get(lmax_apw_);
        lmax_rho_       = parser["parameters"]["lmax_rho"].get(lmax_rho_);
        lmax_pot_       = parser["parameters"]["lmax_pot"].get(lmax_pot_);
        num_mag_dims_   = parser["parameters"]["num_mag_dims"].get(num_mag_dims_);
        auto_rmt_       = parser["parameters"]["auto_rmt"].get(auto_rmt_);
        use_symmetry_   = parser["parameters"]["use_symmetry"].get(use_symmetry_);
        gamma_point_    = parser["parameters"]["gamma_point"].get(gamma_point_);
    }
};

};

#endif // __INPUT_H__


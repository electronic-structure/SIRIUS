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

#include "vector3d.h"
#include "matrix3d.h"
#include "runtime.h"
#include "constants.h"
#include "utils.h"

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
    vector3d<double> a0_;
    vector3d<double> a1_;
    vector3d<double> a2_;

    std::vector<std::string> labels_;
    std::map<std::string, std::string> atom_files_;
    std::vector< std::vector< std::vector<double> > > coordinates_;

    bool exist_{false};

    void read(json const& parser)
    {
        if (parser.count("unit_cell")) {
            exist_ = true;

            auto section = parser["unit_cell"];
            auto a0 = section["lattice_vectors"][0].get<std::vector<double>>();
            auto a1 = section["lattice_vectors"][1].get<std::vector<double>>();
            auto a2 = section["lattice_vectors"][2].get<std::vector<double>>();

            if (a0.size() != 3 || a1.size() != 3 || a2.size() != 3) {
                TERMINATE("wrong lattice vectors");
            }

            double scale = section.value("lattice_vectors_scale", 1.0);

            for (int x: {0, 1, 2}) {
                a0_[x] = a0[x] * scale;
                a1_[x] = a1[x] * scale;
                a2_[x] = a2[x] * scale;
            }

            matrix3d<double> lv;
            for (int x: {0, 1, 2}) {
                lv(x, 0) = a0_[x];
                lv(x, 1) = a1_[x];
                lv(x, 2) = a2_[x];
            }
            auto ilv = inverse(lv);
            
            labels_.clear();
            coordinates_.clear();

            std::string units = section.value("atom_coordinate_units", "lattice");
            
            for (auto& label: section["atom_types"]) {
                if (std::find(std::begin(labels_), std::end(labels_), label) != std::end(labels_)) {
                    TERMINATE("duplicate atom type label");
                }
                labels_.push_back(label);
            }
            
            if (section.count("atom_files")) {
                for (auto& label: labels_) {
                    atom_files_[label] = section["atom_files"].value(label, "");
                }
            }
            
            for (int iat = 0; iat < (int)labels_.size(); iat++) {
                coordinates_.push_back(std::vector< std::vector<double> >());
                for (size_t ia = 0; ia < section["atoms"][labels_[iat]].size(); ia++) {
                    auto v = section["atoms"][labels_[iat]][ia].get<std::vector<double>>();

                    if (!(v.size() == 3 || v.size() == 6)) {
                        TERMINATE("wrong coordinates size");
                    }
                    if (v.size() == 3) {
                        v.resize(6, 0.0);
                    }

                    vector3d<double> v1(v[0], v[1], v[2]);
                    if (units == "A") {
                        for (int x: {0, 1, 2}) {
                            v1[x] /= bohr_radius;
                        }
                    }
                    if (units == "au" || units == "A") {
                        v1 = ilv * v1;
                        auto rv1 = Utils::reduce_coordinates(v1);
                        for (int x: {0, 1, 2}) {
                            v[x] = rv1.first[x];
                        }
                    }

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

    void read(json const& parser)
    {
        if (parser.count("mixer")) {
            exist_ = true;
            auto section = parser["mixer"];
            beta_               = section.value("beta", beta_);
            beta0_              = section.value("beta0", beta0_);
            linear_mix_rms_tol_ = section.value("linear_mix_rms_tol", linear_mix_rms_tol_);
            max_history_        = section.value("max_history", max_history_);
            type_               = section.value("type", type_);
        }
    }
};

/** \todo real-space projectors are not part of iterative solver */
struct Iterative_solver_input_section
{
    std::string type_{""};
    int num_steps_{20};
    int subspace_size_{4};
    double energy_tolerance_{1e-6};
    double residual_tolerance_{1e-6};
    int converge_by_energy_{1}; // TODO: rename, this is meaningless
    int converge_occupied_{0};
    int min_num_res_{0};
    int real_space_prj_{0}; // TODO: move it from here to parameters
    double R_mask_scale_{1.5};
    double mask_alpha_{3};
    int num_singular_{-1};

    void read(json const& parser)
    {
        if (parser.count("iterative_solver")) {
            type_               = parser["iterative_solver"].value("type", type_);
            num_steps_          = parser["iterative_solver"].value("num_steps", num_steps_);
            subspace_size_      = parser["iterative_solver"].value("subspace_size", subspace_size_);
            energy_tolerance_   = parser["iterative_solver"].value("energy_tolerance", energy_tolerance_);
            residual_tolerance_ = parser["iterative_solver"].value("residual_tolerance", residual_tolerance_);
            converge_by_energy_ = parser["iterative_solver"].value("converge_by_energy", converge_by_energy_);
            converge_occupied_  = parser["iterative_solver"].value("converge_occupied", converge_occupied_);
            min_num_res_        = parser["iterative_solver"].value("min_num_res", min_num_res_);
            real_space_prj_     = parser["iterative_solver"].value("real_space_prj", real_space_prj_);
            R_mask_scale_       = parser["iterative_solver"].value("R_mask_scale", R_mask_scale_);
            mask_alpha_         = parser["iterative_solver"].value("mask_alpha", mask_alpha_);
            num_singular_       = parser["iterative_solver"].value("num_singular", num_singular_);
        }
    }
};

/// Parse control input section.
/** The following part of the input file is parsed:
 *  \code{.json}
 *    "control" : {
 *      "mpi_grid_dims" : (1- 2- or 3-dimensional vector<int>) MPI grid layout
 *      "cyclic_block_size" : (int) PBLAS / ScaLAPACK block size
 *      "reduce_gvec" : (bool) use reduced G-vector set (reduce_gvec = true) or full set (reduce_gvec = false)
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
    int cyclic_block_size_{-1};
    bool reduce_gvec_{true};
    std::string std_evp_solver_name_{""};
    std::string gen_evp_solver_name_{""};
    std::string fft_mode_{"serial"};
    std::string processing_unit_{""};
    double rmt_max_{2.2};
    double spglib_tolerance_{1e-4};

    void read(json const& parser)
    {
        mpi_grid_dims_       = parser["control"].value("mpi_grid_dims", mpi_grid_dims_); 
        cyclic_block_size_   = parser["control"].value("cyclic_block_size", cyclic_block_size_);
        std_evp_solver_name_ = parser["control"].value("std_evp_solver_type", std_evp_solver_name_);
        gen_evp_solver_name_ = parser["control"].value("gen_evp_solver_type", gen_evp_solver_name_);
        processing_unit_     = parser["control"].value("processing_unit", processing_unit_);
        fft_mode_            = parser["control"].value("fft_mode", fft_mode_);
        reduce_gvec_         = parser["control"].value("reduce_gvec", reduce_gvec_);
        rmt_max_             = parser["control"].value("rmt_max", rmt_max_);
        spglib_tolerance_    = parser["control"].value("spglib_tolerance", spglib_tolerance_);

        auto strings = {&std_evp_solver_name_, &gen_evp_solver_name_, &fft_mode_, &processing_unit_};
        for (auto s: strings) {
            std::transform(s->begin(), s->end(), s->begin(), ::tolower);
        }
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
    std::vector<int> ngridk_{1, 1, 1};
    std::vector<int> shiftk_{0, 0, 0};
    int num_dft_iter_{100};
    double energy_tol_{1e-5};
    double potential_tol_{1e-5};
    bool molecule_{false};

    void read(json const& parser)
    {
        esm_ = parser["parameters"].value("electronic_structure_method", esm_);
        std::transform(esm_.begin(), esm_.end(), esm_.begin(), ::tolower);

        /* read list of XC functionals */
        if (parser["parameters"].count("xc_functionals")) {
            xc_functionals_.clear();
            for (auto& label: parser["parameters"]["xc_functionals"]) {
                xc_functionals_.push_back(label);
            }
        }

        core_relativity_ = parser["parameters"].value("core_relativity", core_relativity_);
        std::transform(core_relativity_.begin(), core_relativity_.end(), core_relativity_.begin(), ::tolower);

        valence_relativity_ = parser["parameters"].value("valence_relativity", valence_relativity_);
        std::transform(valence_relativity_.begin(), valence_relativity_.end(), valence_relativity_.begin(), ::tolower);

        num_fv_states_  = parser["parameters"].value("num_fv_states", num_fv_states_);
        smearing_width_ = parser["parameters"].value("smearing_width", smearing_width_);
        pw_cutoff_      = parser["parameters"].value("pw_cutoff", pw_cutoff_);
        aw_cutoff_      = parser["parameters"].value("aw_cutoff", aw_cutoff_);
        gk_cutoff_      = parser["parameters"].value("gk_cutoff", gk_cutoff_);
        lmax_apw_       = parser["parameters"].value("lmax_apw", lmax_apw_);
        lmax_rho_       = parser["parameters"].value("lmax_rho", lmax_rho_);
        lmax_pot_       = parser["parameters"].value("lmax_pot", lmax_pot_);
        num_mag_dims_   = parser["parameters"].value("num_mag_dims", num_mag_dims_);
        auto_rmt_       = parser["parameters"].value("auto_rmt", auto_rmt_);
        use_symmetry_   = parser["parameters"].value("use_symmetry", use_symmetry_);
        gamma_point_    = parser["parameters"].value("gamma_point", gamma_point_);
        ngridk_         = parser["parameters"].value("ngridk", ngridk_);
        shiftk_         = parser["parameters"].value("shiftk", shiftk_);
        num_dft_iter_   = parser["parameters"].value("num_dft_iter", num_dft_iter_);
        energy_tol_     = parser["parameters"].value("energy_tol", energy_tol_);
        potential_tol_  = parser["parameters"].value("potential_tol", potential_tol_);
        molecule_       = parser["parameters"].value("molecule", molecule_);
    }
};

};

#endif // __INPUT_H__


// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains input parameters structures.
 */

#ifndef __INPUT_H__
#define __INPUT_H__

#include "constants.h"
#include "sddk.hpp"

using namespace geometry3d;

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
 *          "atom_coordinate_units" : units,
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
 *
 *  The "atom_coordinate_units" string is optional. By default it is set to "lattice" which means that the
 *  atomic coordinates are provided in lattice (fractional) units. It can also be specified in "A" or "au" which
 *  means that the input atomic coordinates are Cartesian and provided in Angstroms or atomic units of length.
 *  This is useful in setting up the molecule calculation.
 */
struct Unit_cell_input
{
    /// First vector of the unit cell.
    vector3d<double> a0_;

    /// Second vector of the unit cell.
    vector3d<double> a1_;

    /// Third vector of the unit cell.
    vector3d<double> a2_;

    /// Labels of the atom types.
    std::vector<std::string> labels_;

    /// Mapping between a label of atom type and corresponding atomic species file.
    std::map<std::string, std::string> atom_files_;

    /// Atomic coordinates.
    /** Outer vector size is equal to the number of atom types. */
    std::vector<std::vector<std::vector<double>>> coordinates_;

    /// True if this section exists in the input file.
    bool exist_{false};

    /// Read the \b unit_cell input section.
    void read(json const& parser)
    {
        if (parser.count("unit_cell")) {
            exist_ = true;

            auto section = parser["unit_cell"];
            auto a0      = section["lattice_vectors"][0].get<std::vector<double>>();
            auto a1      = section["lattice_vectors"][1].get<std::vector<double>>();
            auto a2      = section["lattice_vectors"][2].get<std::vector<double>>();

            if (a0.size() != 3 || a1.size() != 3 || a2.size() != 3) {
                TERMINATE("wrong lattice vectors");
            }

            double scale = section.value("lattice_vectors_scale", 1.0);

            for (int x : {0, 1, 2}) {
                a0_[x] = a0[x] * scale;
                a1_[x] = a1[x] * scale;
                a2_[x] = a2[x] * scale;
            }

            matrix3d<double> lv;
            for (int x : {0, 1, 2}) {
                lv(x, 0) = a0_[x];
                lv(x, 1) = a1_[x];
                lv(x, 2) = a2_[x];
            }
            auto ilv = inverse(lv);

            labels_.clear();
            coordinates_.clear();

            std::string units = section.value("atom_coordinate_units", "lattice");

            for (auto& label : section["atom_types"]) {
                if (std::find(std::begin(labels_), std::end(labels_), label) != std::end(labels_)) {
                    TERMINATE("duplicate atom type label");
                }
                labels_.push_back(label);
            }

            if (section.count("atom_files")) {
                for (auto& label : labels_) {
                    atom_files_[label] = section["atom_files"].value(label, "");
                }
            }

            for (int iat = 0; iat < (int)labels_.size(); iat++) {
                coordinates_.push_back(std::vector<std::vector<double>>());
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
                        for (int x : {0, 1, 2}) {
                            v1[x] /= bohr_radius;
                        }
                    }
                    if (units == "au" || units == "A") {
                        v1       = ilv * v1;
                        auto rv1 = reduce_coordinates(v1);
                        for (int x : {0, 1, 2}) {
                            v[x] = rv1.first[x];
                        }
                    }

                    coordinates_[iat].push_back(v);
                }
            }
        }
    }
};

/// Parse mixer input section.
struct Mixer_input
{
    /// Mixing paramter.
    double beta_{0.7};

    /// Mixin ratio in case of initial linear mixing.
    double beta0_{0.15};

    /// RMS tolerance above which the linear mixing is triggered.
    double linear_mix_rms_tol_{1e6};

    /// Type of the mixer.
    /** Available types are: "broyden1", "broyden2", "linear" */
    std::string type_{"broyden1"};

    /// Number of history steps for Broyden-type mixers.
    int max_history_{8};

    /// True if this section exists in the input file.
    bool exist_{false};

    /// Read the \b mixer input section.
    void read(json const& parser)
    {
        if (parser.count("mixer")) {
            exist_              = true;
            auto section        = parser["mixer"];
            beta_               = section.value("beta", beta_);
            beta0_              = section.value("beta0", beta0_);
            linear_mix_rms_tol_ = section.value("linear_mix_rms_tol", linear_mix_rms_tol_);
            max_history_        = section.value("max_history", max_history_);
            type_               = section.value("type", type_);
        }
    }
};

/** \todo real-space projectors are not part of iterative solver */
struct Iterative_solver_input
{
    /// Type of the iterative solver.
    std::string type_{""};

    /// Number of steps (iterations) of the solver.
    int num_steps_{20};

    /// Size of the variational subspace is this number times the number of bands.
    int subspace_size_{4};

    /// Tolerance for the eigen-energy difference \f$ |\epsilon_i^{old} - \epsilon_i^{new} | \f$.
    /** This parameter is reduced during the SCF cycle to reach the high accuracy of the wave-functions. */
    double energy_tolerance_{1e-2};

    /// Tolerance for the residual L2 norm.
    double residual_tolerance_{1e-6};

    /// Additional tolerance for empty states.
    /** Setting this variable to 0 will treat empty states with the same tolerance as occupied states. */
    double empty_states_tolerance_{1e-5};

    /// Defines the flavour of the iterative solver.
    /** If converge_by_energy is set to 0, then the residuals are estimated by their norm. If converge_by_energy
     *  is set to 1 then the residuals are estimated by the eigen-energy difference. This allows to estimate the
     *  unconverged residuals and only then compute only the unconverged. */
    int converge_by_energy_{1}; // TODO: rename, this is meaningless

    /// Minimum number of residuals to continue iterative diagonalization process.
    int min_num_res_{0};

    int real_space_prj_{0}; // TODO: move it from here to parameters
    double R_mask_scale_{1.5};
    double mask_alpha_{3};

    /// Number of singular components for the LAPW Davidson solver.
    int num_singular_{-1};

    /// Control the subspace expansion.
    /** If true, keep basis orthogonal and solve standard eigen-value problem. If false, add preconditioned residuals
     *  as they are and solve generalized eigen-value problem. */
    bool orthogonalize_{true};

    bool init_eval_old_{true};

    /// Tell how to initialize the subspace.
    /** It can be either "lcao", i.e. start from the linear combination of atomic orbitals or "random" –- start from
     *  the randomized wave functions. */
    std::string init_subspace_{"lcao"};

    void read(json const& parser)
    {
        if (parser.count("iterative_solver")) {
            type_                   = parser["iterative_solver"].value("type", type_);
            num_steps_              = parser["iterative_solver"].value("num_steps", num_steps_);
            subspace_size_          = parser["iterative_solver"].value("subspace_size", subspace_size_);
            energy_tolerance_       = parser["iterative_solver"].value("energy_tolerance", energy_tolerance_);
            residual_tolerance_     = parser["iterative_solver"].value("residual_tolerance", residual_tolerance_);
            empty_states_tolerance_ = parser["iterative_solver"].value("empty_states_tolerance", empty_states_tolerance_);
            converge_by_energy_     = parser["iterative_solver"].value("converge_by_energy", converge_by_energy_);
            min_num_res_            = parser["iterative_solver"].value("min_num_res", min_num_res_);
            real_space_prj_         = parser["iterative_solver"].value("real_space_prj", real_space_prj_);
            R_mask_scale_           = parser["iterative_solver"].value("R_mask_scale", R_mask_scale_);
            mask_alpha_             = parser["iterative_solver"].value("mask_alpha", mask_alpha_);
            num_singular_           = parser["iterative_solver"].value("num_singular", num_singular_);
            orthogonalize_          = parser["iterative_solver"].value("orthogonalize", orthogonalize_);
            init_eval_old_          = parser["iterative_solver"].value("init_eval_old", init_eval_old_);
            init_subspace_          = parser["iterative_solver"].value("init_subspace", init_subspace_);
            std::transform(init_subspace_.begin(), init_subspace_.end(), init_subspace_.begin(), ::tolower);
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
struct Control_input
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
    /// Level of verbosity.
    /** The following convention in proposed:
     *    - 0: silent mode (no output is printed) \n
     *    - 1: basic output (low level of output) \n
     *    - 2: extended output (medium level of output) \n
     *    - 3: extensive output (hi level of output) */
    int verbosity_{0};
    int verification_{0};
    int num_bands_to_print_{10};
    bool print_performance_{false};
    bool print_memory_usage_{false};
    bool print_checksum_{false};
    bool print_hash_{false};
    bool print_stress_{false};
    bool print_forces_{false};
    bool print_timers_{true};
    bool print_neighbors_{false};

    void read(json const& parser)
    {
        if (parser.count("control")) {
            mpi_grid_dims_       = parser["control"].value("mpi_grid_dims", mpi_grid_dims_);
            cyclic_block_size_   = parser["control"].value("cyclic_block_size", cyclic_block_size_);
            std_evp_solver_name_ = parser["control"].value("std_evp_solver_type", std_evp_solver_name_);
            gen_evp_solver_name_ = parser["control"].value("gen_evp_solver_type", gen_evp_solver_name_);
            processing_unit_     = parser["control"].value("processing_unit", processing_unit_);
            fft_mode_            = parser["control"].value("fft_mode", fft_mode_);
            reduce_gvec_         = parser["control"].value("reduce_gvec", reduce_gvec_);
            rmt_max_             = parser["control"].value("rmt_max", rmt_max_);
            spglib_tolerance_    = parser["control"].value("spglib_tolerance", spglib_tolerance_);
            verbosity_           = parser["control"].value("verbosity", verbosity_);
            verification_        = parser["control"].value("verification", verification_);
            num_bands_to_print_  = parser["control"].value("num_bands_to_print", num_bands_to_print_);
            print_performance_   = parser["control"].value("print_performance", print_performance_);
            print_memory_usage_  = parser["control"].value("print_memory_usage", print_memory_usage_);
            print_checksum_      = parser["control"].value("print_checksum", print_checksum_);
            print_hash_          = parser["control"].value("print_hash", print_hash_);
            print_stress_        = parser["control"].value("print_stress", print_stress_);
            print_forces_        = parser["control"].value("print_forces", print_forces_);
            print_timers_        = parser["control"].value("print_timers", print_timers_);
            print_neighbors_     = parser["control"].value("print_neighbors", print_neighbors_);

            auto strings = {&std_evp_solver_name_, &gen_evp_solver_name_, &fft_mode_, &processing_unit_};
            for (auto s : strings) {
                std::transform(s->begin(), s->end(), s->begin(), ::tolower);
            }
        }
    }
};

struct Parameters_input
{
    /// Electronic structure method.
    std::string electronic_structure_method_{"none"};

    std::vector<std::string> xc_functionals_;
    std::string core_relativity_{"dirac"};
    std::string valence_relativity_{"zora"};

    /// Number of bands.
    /** In spin-collinear case this is the number of bands for each spin channel. */
    int num_bands_{-1};

    /// Number of first-variational states.
    int num_fv_states_{-1};

    /// Smearing function width.
    double smearing_width_{0.01}; // in Ha

    /// Cutoff for plane-waves (for density and potential expansion).
    double pw_cutoff_{20.0}; // in a.u.^-1

    /// Cutoff for augmented-wave functions.
    double aw_cutoff_{7.0}; // this is R_{MT} * |G+k|_{max}

    /// Cutoff for |G+k| plane-waves.
    double gk_cutoff_{6.0}; // in a.u.^-1

    /// Maximum l for APW functions.
    int lmax_apw_{8};

    /// Maximum l for density.
    int lmax_rho_{8};

    /// Maximum l for potential
    int lmax_pot_{8};

    /// Number of dimensions of the magnetization and effective magnetic field (0, 1 or 3).
    int num_mag_dims_{0};

    /// Scale muffin-tin radii automatically.
    int auto_rmt_{1};

    std::vector<int> ngridk_{1, 1, 1};
    std::vector<int> shiftk_{0, 0, 0};
    int num_dft_iter_{100};
    double energy_tol_{1e-5};
    double potential_tol_{1e-5};

    /// True if this is a molecule calculation.
    bool molecule_{false};

    /// True if gamma-point (real) version of the PW code is used.
    bool gamma_point_{false};

    /// True if spin-orbit correction is applied.
    bool so_correction_{false};

    /// True if Hubbard (or U) correction is applied.
    bool hubbard_correction_{false};

    /// True if symmetry is used.
    bool use_symmetry_{true};

    double nn_radius_{-1};

    /// Effective screening medium.
    bool enable_esm_{false};

    /// Type of periodic boundary conditions.
    std::string esm_bc_{"pbc"};

    void read(json const& parser)
    {
        if (parser.count("parameters")) {
            electronic_structure_method_ = parser["parameters"].value("electronic_structure_method",
                                                                      electronic_structure_method_);
            std::transform(electronic_structure_method_.begin(), electronic_structure_method_.end(),
                           electronic_structure_method_.begin(), ::tolower);

            /* read list of XC functionals */
            if (parser["parameters"].count("xc_functionals")) {
                xc_functionals_.clear();
                for (auto& label : parser["parameters"]["xc_functionals"]) {
                    xc_functionals_.push_back(label);
                }
            }

            core_relativity_ = parser["parameters"].value("core_relativity", core_relativity_);
            std::transform(core_relativity_.begin(), core_relativity_.end(), core_relativity_.begin(), ::tolower);

            valence_relativity_ = parser["parameters"].value("valence_relativity", valence_relativity_);
            std::transform(valence_relativity_.begin(), valence_relativity_.end(), valence_relativity_.begin(),
                           ::tolower);

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
            nn_radius_      = parser["parameters"].value("nn_radius", nn_radius_);
            if (parser["parameters"].count("spin_orbit")) {
                so_correction_ = parser["parameters"].value("spin_orbit", so_correction_);

                // check that the so correction is actually needed. the
                // parameter spin_orbit can still be indicated to false
                if (so_correction_) {
                    num_mag_dims_  = 3;
                }
            }

            if (parser["parameters"].count("hubbard_correction")) {
                hubbard_correction_ = parser["parameters"].value("hubbard_correction", hubbard_correction_);
            }

        }
    }
};

/// Settings control the internal parameters related to the numerical implementation.
struct Settings_input
{
    /// Number of points (per a.u.^-1) for radial integral interpolation for local part of pseudopotential.
    int nprii_vloc_{200};
    int nprii_beta_{20};
    int nprii_aug_{20};
    int nprii_rho_core_{20};
    bool always_update_wf_{true};
    double mixer_rss_min_{1e-12};

    void read(json const& parser)
    {
        if (parser.count("settings")) {
            nprii_vloc_       = parser["settings"].value("nprii_vloc", nprii_vloc_);
            nprii_beta_       = parser["settings"].value("nprii_beta", nprii_beta_);
            nprii_aug_        = parser["settings"].value("nprii_aug", nprii_aug_);
            nprii_rho_core_   = parser["settings"].value("nprii_rho_core", nprii_rho_core_);
            always_update_wf_ = parser["settings"].value("always_update_wf", always_update_wf_);
            mixer_rss_min_    = parser["settings"].value("mixer_rss_min", mixer_rss_min_);
        }
    }
};

struct Hubbard_input
{
    int number_of_species{1};
    bool hubbard_correction_{false};
    bool simplified_hubbard_correction_{false};
    bool orthogonalize_hubbard_orbitals_{false};
    bool normalize_hubbard_orbitals_{false};
    bool hubbard_starting_magnetization_{false};
    bool hubbard_U_plus_V_{false};
    int projection_method_{0};
    std::string wave_function_file_;
    std::vector<std::pair<std::string, std::vector<double>>> species;

    bool hubbard_correction() const
    {
        return hubbard_correction_;
    }

    void read(json const& parser)
    {
        if (!parser.count("hubbard"))
            return;

        orthogonalize_hubbard_orbitals_ = false;
        if (parser["hubbard"].count("orthogonalize_hubbard_wave_functions")) {
            orthogonalize_hubbard_orbitals_ = parser["hubbard"].value("orthogonalize_hubbard_wave_functions", orthogonalize_hubbard_orbitals_);
        }

        normalize_hubbard_orbitals_ = false;
        if (parser["hubbard"].count("normalize_hubbard_wave_functions")) {
            normalize_hubbard_orbitals_ = parser["hubbard"].value("normalize_hubbard_wave_functions", normalize_hubbard_orbitals_);
        }

        if (parser["hubbard"].count("simplified_hubbard_correction")) {
            simplified_hubbard_correction_ = parser["hubbard"].value("simplified_hubbard_correction",
                                                                     simplified_hubbard_correction_);
        }
        std::vector<double> coef_;
        std::vector<std::string> labels_;
        coef_.clear();
        coef_.resize(9, 0.0);
        species.clear();
        labels_.clear();

        for (auto& label : parser["unit_cell"]["atom_types"]) {
            if (std::find(std::begin(labels_), std::end(labels_), label) != std::end(labels_)) {
                TERMINATE("duplicate atom type label");
            }
            labels_.push_back(label);
        }

        // by default we use the atomic orbitals given in the pseudo potentials
        this->projection_method_ = 0;

        if (parser["hubbard"].count("projection_method")) {
            std::string projection_method__ = parser["hubbard"]["projection_method"].get<std::string>();
            if (projection_method__ == "file") {
                // they are provided by a external file
                if (parser["hubbard"].count("wave_function_file")) {
                    this->wave_function_file_ = parser["hubbard"]["wave_function_file"].get<std::string>();
                    this->projection_method_ = 1;
                } else {
                    TERMINATE("The hubbard projection method 'file' requires the option 'wave_function_file' to be defined");
                }
            }

            if (projection_method__ == "pseudo") {
                this->projection_method_ = 2;
            }
        }

        for (auto &label : labels_) {
            for(size_t d = 0; d < coef_.size(); d++)
                coef_[d] = 0.0;

            if(parser["hubbard"][label].count("U")) {
                coef_[0] = parser["hubbard"][label]["U"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("J")) {
                coef_[1] = parser["hubbard"][label]["J"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("B")) {
                coef_[2] = parser["hubbard"][label]["B"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("E2")) {
                coef_[2] = parser["hubbard"][label]["E2"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("E3")) {
                coef_[3] = parser["hubbard"][label]["E3"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("alpha")) {
                coef_[4] = parser["hubbard"][label]["alpha"].get<double>();
                hubbard_correction_ = true;
            }

            if(parser["hubbard"][label].count("beta")) {
                coef_[5] = parser["hubbard"][label]["beta"].get<double>();
                hubbard_correction_ = true;
            }

            // angle for the starting magnetization in deg, convert it
            // in radian

            if(parser["hubbard"][label].count("starting_magnetization")) {
                coef_[6] = parser["hubbard"][label]["starting_magnetization"].get<double>();
                hubbard_starting_magnetization_ = true;
            }

            if(parser["hubbard"][label].count("starting_magnetization_theta_angle")) {
                coef_[7] = parser["hubbard"][label]["starting_magnetization_theta_angle"].get<double>() * M_PI / 180.0 ;
                hubbard_starting_magnetization_ = true;
            }

            if(parser["hubbard"][label].count("starting_magnetization_phi_angle")) {
                coef_[8] = parser["hubbard"][label]["starting_magnetization_phi_angle"].get<double>() * M_PI/ 180.0 ;
                hubbard_starting_magnetization_ = true;
            }

            // now convert eV in Ha
            for (int s = 0; s < static_cast<int>(coef_.size() - 3); s++) {
                coef_[s] /= ha2ev;
            }

            species.push_back(std::make_pair(label, coef_));
        }
        if (parser["hubbard"].count("hubbard_u_plus_v")) {
            hubbard_U_plus_V_ = true;
        }


        if (!hubbard_correction_) {
            TERMINATE("The hubbard section is empty");
        }

    }
};
};

#endif // __INPUT_H__

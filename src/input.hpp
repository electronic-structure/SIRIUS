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

/** \file input.hpp
 *
 *  \brief Contains input parameters structures.
 *
 *  \todo Some of the parameters belong to SCF ground state mini-app. Mini-app should parse this values itself.
 *  \todo parse atomic coordinates and magnetic field separtely, not as 6D vector.
 */

#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <list>
#include "constants.hpp"
#include "SDDK/geometry3d.hpp"
#include "utils/json.hpp"

using namespace geometry3d;
using namespace nlohmann;

namespace sirius {

/// Parse unit cell input section.
/** The following part of the input file is parsed:
    \code{.json}
    "unit_cell" : {
        "lattice_vectors" : [
            [a1_x, a1_y, a1_z],
            [a2_x, a2_y, a2_z],
            [a3_x, a3_y, a3_z]
        ],

        "lattice_vectors_scale" : (float) scale,

        "atom_types" : ["label_A", "label_B", ...],

        "atom_files" : {
            "label_A" : "file_A",
            "label_B" : "file_B",
            ...
        },

        "atom_coordinate_units" : units,

        "atoms" : {
            "label_A": [
                coordinates_A_1,
                coordinates_A_2,
                ...
            ],
            "label_B" : [
                coordinates_B_1,
                coordinates_B_2,
                ...
            ]
        }
    }
    \endcode

    The "atom_coordinate_units" string is optional. By default it is assumed to be "lattice" which means that the
    atomic coordinates are provided in lattice (fractional) units. It can also be specified in "A" or "au" which
    means that the input atomic coordinates are Cartesian and provided in Angstroms or atomic units of length.
    This is useful in setting up the molecule calculation.
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

            auto a0 = section["lattice_vectors"][0].get<std::vector<double>>();
            auto a1 = section["lattice_vectors"][1].get<std::vector<double>>();
            auto a2 = section["lattice_vectors"][2].get<std::vector<double>>();

            if (a0.size() != 3 || a1.size() != 3 || a2.size() != 3) {
                throw std::runtime_error("wrong lattice vectors");
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
                    throw std::runtime_error("duplicate atom type label");
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
                        throw std::runtime_error("wrong coordinates size");
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
/** The following part of the input file is parsed:
    \code{.json}
    "mixer" : {
      "beta" : (float) beta,
      "beta0" : beta0,
      "linear_mix_rms_tol" : 
    }
    \endcode
 */
struct Mixer_input
{
    /// Mixing paramter.
    double beta_{0.7};

    /// Mixing ratio in case of initial linear mixing.
    double beta0_{0.15};

    /// RMS tolerance above which the linear mixing is triggered.
    double linear_mix_rms_tol_{1e6};

    /// Type of the mixer.
    /** Available types are: "broyden1", "broyden2", "linear" */
    std::string type_{"broyden1"};

    /// Number of history steps for Broyden-type mixers.
    int max_history_{8};

    /// Scaling factor for mixing parameter.
    double beta_scaling_factor_{1};

    /// Use Hartree potential in the inner() product for residuals.
    bool use_hartree_{false};

    /// True if this section exists in the input file.
    bool exist_{false};

    /// Read the \b mixer input section.
    void read(json const& parser)
    {
        if (parser.count("mixer")) {
            exist_               = true;
            auto section         = parser["mixer"];
            beta_                = section.value("beta", beta_);
            beta0_               = section.value("beta0", beta0_);
            linear_mix_rms_tol_  = section.value("linear_mix_rms_tol", linear_mix_rms_tol_);
            max_history_         = section.value("max_history", max_history_);
            type_                = section.value("type", type_);
            beta_scaling_factor_ = section.value("beta_scaling_factor", beta_scaling_factor_);
            use_hartree_         = section.value("use_hartree", use_hartree_);
        }
    }
};

/// Parse the parameters of iterative solver.
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
    double empty_states_tolerance_{0};

    /// Defines the flavour of the iterative solver.
    /** If converge_by_energy is set to 0, then the residuals are estimated by their norm. If converge_by_energy
        is set to 1 then the residuals are estimated by the eigen-energy difference. This allows to estimate the
        unconverged residuals and then compute only the unconverged ones.
     */
    int converge_by_energy_{1}; // TODO: rename, this is meaningless

    /// Minimum number of residuals to continue iterative diagonalization process.
    int min_num_res_{0};

    /// Number of singular components for the LAPW Davidson solver.
    int num_singular_{-1};

    /// Control the subspace expansion.
    /** If true, keep basis orthogonal and solve standard eigen-value problem. If false, add preconditioned residuals
        as they are and solve generalized eigen-value problem.
     */
    bool orthogonalize_{true};

    /// Initialize eigen-values with previous (old) values.
    bool init_eval_old_{true};

    /// Tell how to initialize the subspace.
    /** It can be either "lcao", i.e. start from the linear combination of atomic orbitals or "random" –- start from
        the randomized wave functions.
     */
    std::string init_subspace_{"lcao"};

    void read(json const& parser)
    {
        if (parser.count("iterative_solver")) {
            auto section            = parser["iterative_solver"];
            type_                   = section.value("type", type_);
            num_steps_              = section.value("num_steps", num_steps_);
            subspace_size_          = section.value("subspace_size", subspace_size_);
            energy_tolerance_       = section.value("energy_tolerance", energy_tolerance_);
            residual_tolerance_     = section.value("residual_tolerance", residual_tolerance_);
            empty_states_tolerance_ = section.value("empty_states_tolerance", empty_states_tolerance_);
            converge_by_energy_     = section.value("converge_by_energy", converge_by_energy_);
            min_num_res_            = section.value("min_num_res", min_num_res_);
            num_singular_           = section.value("num_singular", num_singular_);
            orthogonalize_          = section.value("orthogonalize", orthogonalize_);
            init_eval_old_          = section.value("init_eval_old", init_eval_old_);
            init_subspace_          = section.value("init_subspace", init_subspace_);
            std::transform(init_subspace_.begin(), init_subspace_.end(), init_subspace_.begin(), ::tolower);
        }
    }
};

/// Parse control input section.
/** The following part of the input file is parsed:
    \code{.json}
    "control" : {
      "mpi_grid_dims" : (1- 2- or 3-dimensional vector<int>) MPI grid layout
      "cyclic_block_size" : (int) PBLAS / ScaLAPACK block size
      "reduce_gvec" : (bool) use reduced G-vector set (reduce_gvec = true) or full set (reduce_gvec = false)
      "std_evp_solver_type" : (string) type of eigen-solver for the standard eigen-problem
      "gen_evp_solver_type" : (string) type of eigen-solver for the generalized eigen-problem
      "processing_unit" : (string) primary processing unit
      "fft_mode" : (string) serial or parallel FFT
    }
    \endcode
    Parameters of the control input sections do not in general change the numerics, but instead control how the
    results are obtained. Changing parameters in control section should not change the significant digits in final
    results.
 */
struct Control_input
{
    /// Dimensions of the MPI grid (if used).
    std::vector<int> mpi_grid_dims_;

    /// Block size for ScaLAPACK and ELPA.
    int cyclic_block_size_{-1};

    /// Reduce G-vectors by inversion symmetry.
    /** For real-valued functions like density and potential it is sufficient to store only half of the G-vectors
     *  and use the relation f(G) = f^{*}(-G) to recover second half of the plane-wave expansion coefficients. */
    bool reduce_gvec_{true};

    /// Standard eigen-value solver to use.
    std::string std_evp_solver_name_{""};

    /// Generalized eigen-value solver to use.
    std::string gen_evp_solver_name_{""};

    /// Coarse grid FFT mode ("serial" or "parallel").
    std::string fft_mode_{"serial"};

    /// Main processing unit to run on.
    std::string processing_unit_{""};

    /// Maximum allowed muffin-tin radius in case of LAPW.
    double rmt_max_{2.2};

    /// Tolerance of the spglib in finding crystal symmetries.
    double spglib_tolerance_{1e-4};

    /// Level of verbosity.
    /** The following convention in proposed:
     *    - 0: silent mode (no output is printed) \n
     *    - 1: basic output (low level of output) \n
     *    - 2: extended output (medium level of output) \n
     *    - 3: extensive output (high level of output) */
    int verbosity_{0};

    /// Level of internal verification.
    int verification_{0};

    /// Number of eigen-values that are printed to the standard output.
    int num_bands_to_print_{10};

    /// If true then performance of some compute-intensive kernels will be printed to the standard output.
    bool print_performance_{false};

    /// If true then memory usage will be printed to the standard output.
    bool print_memory_usage_{false};

    /// If true then the checksums of some arrays will be printed (useful during debug).
    bool print_checksum_{false};

    /// If true then the hashsums of some arrays will be printed.
    bool print_hash_{false};

    /// If true then the stress tensor components are printed at the end of SCF run.
    bool print_stress_{false};

    /// If true then the atomic forces are printed at the end of SCF run.
    bool print_forces_{false};

    /// If true then the timer statistics is printed at the end of SCF run.
    bool print_timers_{true};

    /// If true then the list of nearest neighbours for each atom is printed to the standard output.
    bool print_neighbors_{false};

    /// True if second-variational diagonalization is used in LAPW method.
    bool use_second_variation_{true};

    /// Control the usage of the GPU memory.
    /** Possible values are: "low", "medium" and "high". */
    std::string memory_usage_{"high"};

    /// Number of atoms in the beta-projectors chunk.
    int beta_chunk_size_{256};

    void read(json const& parser)
    {
        if (parser.count("control")) {
            auto section         = parser["control"];
            mpi_grid_dims_       = section.value("mpi_grid_dims", mpi_grid_dims_);
            cyclic_block_size_   = section.value("cyclic_block_size", cyclic_block_size_);
            std_evp_solver_name_ = section.value("std_evp_solver_type", std_evp_solver_name_);
            gen_evp_solver_name_ = section.value("gen_evp_solver_type", gen_evp_solver_name_);
            processing_unit_     = section.value("processing_unit", processing_unit_);
            fft_mode_            = section.value("fft_mode", fft_mode_);
            reduce_gvec_         = section.value("reduce_gvec", reduce_gvec_);
            rmt_max_             = section.value("rmt_max", rmt_max_);
            spglib_tolerance_    = section.value("spglib_tolerance", spglib_tolerance_);
            verbosity_           = section.value("verbosity", verbosity_);
            verification_        = section.value("verification", verification_);
            num_bands_to_print_  = section.value("num_bands_to_print", num_bands_to_print_);
            print_performance_   = section.value("print_performance", print_performance_);
            print_memory_usage_  = section.value("print_memory_usage", print_memory_usage_);
            print_checksum_      = section.value("print_checksum", print_checksum_);
            print_hash_          = section.value("print_hash", print_hash_);
            print_stress_        = section.value("print_stress", print_stress_);
            print_forces_        = section.value("print_forces", print_forces_);
            print_timers_        = section.value("print_timers", print_timers_);
            print_neighbors_     = section.value("print_neighbors", print_neighbors_);
            memory_usage_        = section.value("memory_usage", memory_usage_);
            beta_chunk_size_     = section.value("beta_chunk_size", beta_chunk_size_);

            auto strings = {&std_evp_solver_name_, &gen_evp_solver_name_, &fft_mode_, &processing_unit_,
                            &memory_usage_};
            for (auto s : strings) {
                std::transform(s->begin(), s->end(), s->begin(), ::tolower);
            }

            std::list<std::string> kw;
            kw = {"low", "medium", "high"};
            if (std::find(kw.begin(), kw.end(), memory_usage_) == kw.end()) {
                throw std::runtime_error("wrong memory_usage input");
            }
        }
    }
};

/// Parse parameters input section.
/** Most of this parameters control the behavior of sirius::DFT_ground_state class. */
struct Parameters_input
{
    /// Electronic structure method.
    std::string electronic_structure_method_{"none"};

    /// List of XC functions (typically contains exchange term and correlation term).
    std::vector<std::string> xc_functionals_;

    /// Type of core-states relativity in full-potential LAPW case.
    std::string core_relativity_{"dirac"};

    /// Type of valence states relativity in full-potential LAPW case.
    std::string valence_relativity_{"zora"};

    /// Number of bands.
    /** In spin-collinear case this is the number of bands for each spin channel. */
    int num_bands_{-1};

    /// Number of first-variational states.
    int num_fv_states_{-1};

    /// Width of Gaussian smearing function in the units of [Ha].
    double smearing_width_{0.01};

    /// Cutoff for plane-waves (for density and potential expansion) in the units of [a.u.^-1].
    double pw_cutoff_{0.0};

    /// Cutoff for augmented-wave functions.
    double aw_cutoff_{0.0}; // this is R_{MT} * |G+k|_{max}

    /// Cutoff for |G+k| plane-waves in the units of [a.u.^-1].
    double gk_cutoff_{0.0};

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

    /// Regular k-point grid for the SCF ground state.
    std::vector<int> ngridk_{1, 1, 1};

    /// Shift in the k-point grid.
    std::vector<int> shiftk_{0, 0, 0};

    /// Number of SCF iterations.
    int num_dft_iter_{100};

    /// Tolerance in total energy change (in units of [Ha]).
    double energy_tol_{1e-8};

    /// Tolerance for the density root mean square (in units of [a.u.^-3]).
    /** RMS is computed as Sqrt( (1/Omega) \int \delta rho(r) \delta rho(r) dr) */
    double density_tol_{1e-8};

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

    /// Radius of atom nearest-neighbour cluster.
    double nn_radius_{-1};

    /// Effective screening medium.
    bool enable_esm_{false};

    /// Type of periodic boundary conditions.
    std::string esm_bc_{"pbc"};

    /// Reduction of the auxiliary magnetic field at each SCF step.
    double reduce_aux_bf_{0.0};

    /// Introduce extra charge to the system. Positive charge means extra holes, negative charge - extra electrons.
    double extra_charge_{0.0};

    void read(json const& parser)
    {
        if (parser.count("parameters")) {
            auto section = parser["parameters"];
            electronic_structure_method_ = section.value("electronic_structure_method", electronic_structure_method_);
            std::transform(electronic_structure_method_.begin(), electronic_structure_method_.end(),
                           electronic_structure_method_.begin(), ::tolower);
            xc_functionals_.clear();
            /* read list of XC functionals */
            if (section.count("xc_functionals")) {
                xc_functionals_.clear();
                for (auto& label : section["xc_functionals"]) {
                    xc_functionals_.push_back(label);
                }
            }

            if (section.count("vdw_functionals")) {
                xc_functionals_.push_back(section["vdw_functionals"].get<std::string>());
            }

            core_relativity_ = section.value("core_relativity", core_relativity_);
            std::transform(core_relativity_.begin(), core_relativity_.end(), core_relativity_.begin(), ::tolower);

            valence_relativity_ = section.value("valence_relativity", valence_relativity_);
            std::transform(valence_relativity_.begin(), valence_relativity_.end(), valence_relativity_.begin(),
                           ::tolower);

            num_fv_states_  = section.value("num_fv_states", num_fv_states_);
            smearing_width_ = section.value("smearing_width", smearing_width_);
            pw_cutoff_      = section.value("pw_cutoff", pw_cutoff_);
            aw_cutoff_      = section.value("aw_cutoff", aw_cutoff_);
            gk_cutoff_      = section.value("gk_cutoff", gk_cutoff_);
            lmax_apw_       = section.value("lmax_apw", lmax_apw_);
            lmax_rho_       = section.value("lmax_rho", lmax_rho_);
            lmax_pot_       = section.value("lmax_pot", lmax_pot_);
            num_mag_dims_   = section.value("num_mag_dims", num_mag_dims_);
            auto_rmt_       = section.value("auto_rmt", auto_rmt_);
            use_symmetry_   = section.value("use_symmetry", use_symmetry_);
            gamma_point_    = section.value("gamma_point", gamma_point_);
            ngridk_         = section.value("ngridk", ngridk_);
            shiftk_         = section.value("shiftk", shiftk_);
            num_dft_iter_   = section.value("num_dft_iter", num_dft_iter_);
            energy_tol_     = section.value("energy_tol", energy_tol_);
            /* potential_tol is obsolete */
            density_tol_    = section.value("potential_tol", density_tol_);
            density_tol_    = section.value("density_tol", density_tol_);
            molecule_       = section.value("molecule", molecule_);
            nn_radius_      = section.value("nn_radius", nn_radius_);
            reduce_aux_bf_  = section.value("reduce_aux_bf", reduce_aux_bf_);
            extra_charge_   = section.value("extra_charge", extra_charge_);

            if (section.count("spin_orbit")) {
                so_correction_ = section.value("spin_orbit", so_correction_);

                /* spin-orbit correction requires non-collinear magnetism */
                if (so_correction_) {
                    num_mag_dims_ = 3;
                }
            }

            if (section.count("hubbard_correction")) {
                hubbard_correction_ = section.value("hubbard_correction", hubbard_correction_);
            }
        }
    }
};

/// Settings control the internal parameters related to the numerical implementation.
/** Changing of setting parameters will have an impact on the final result. */
struct Settings_input
{
    /// Point density (in a.u.^-1) for interpolating radial integrals of local part of pseudopotential.
    int nprii_vloc_{200};
    /// Point density (in a.u.^-1) for interpolating radial integrals of beta projectors.
    int nprii_beta_{20};
    int nprii_aug_{20};
    int nprii_rho_core_{20};
    bool always_update_wf_{true};

    /// Minimum value of allowed RMS for the mixer.
    /** Mixer will not mix functions if the RMS between previous and current functions is below this tolerance. */
    double mixer_rms_min_{1e-16};

    /// Minimum tolerance of the iterative solver.
    double itsol_tol_min_{1e-13};

    /// Minimum occupancy below which the band is treated as being "empty".
    double min_occupancy_{1e-14};

    /// Fine control of the empty states tolerance.
    /** This is the ratio between the tolerance of empty and occupied states. Used in the code like this:
        \code{.cpp}
        // tolerance of occupied bands
        double tol = ctx_.iterative_solver_tolerance();
        // final tolerance of empty bands
        double empy_tol = std::max(tol * ctx_.settings().itsol_tol_ratio_, itso.empty_states_tolerance_);
        \endcode
    */
    double itsol_tol_ratio_{0};

    /// Scaling parameters of the iterative  solver tolerance.
    /** First number is the scaling of density RMS, that gives the estimate of the new tolerance. Second number is
        the scaling of the old tolerance. New tolerance is then the minimum between the two. This is how it is
        done in the code:
        \code{.cpp}
        double old_tol = ctx_.iterative_solver_tolerance();
        // estimate new tolerance of iterative solver
        double tol = std::min(ctx_.settings().itsol_tol_scale_[0] * rms, ctx_.settings().itsol_tol_scale_[1] * old_tol);
        tol = std::max(ctx_.settings().itsol_tol_min_, tol);
        // set new tolerance of iterative solver
        ctx_.iterative_solver_tolerance(tol);
        \endcode
     */
    std::array<double, 2> itsol_tol_scale_{{0.001, 0.5}};

    double auto_enu_tol_{0};

    /// Initial dimenstions for the fine-grain FFT grid.
    std::array<int, 3> fft_grid_size_{{0, 0, 0}};

    /// Default radial grid for LAPW species.
    std::string radial_grid_{"exponential, 1.0"};

    /// Coverage of sphere in case of spherical harmonics transformation.
    /** 0 is Lebedev-Laikov coverage, 1 is unifrom coverage */
    int sht_coverage_{0};

    void read(json const& parser)
    {
        if (parser.count("settings")) {
            auto section      = parser["settings"];
            nprii_vloc_       = section.value("nprii_vloc", nprii_vloc_);
            nprii_beta_       = section.value("nprii_beta", nprii_beta_);
            nprii_aug_        = section.value("nprii_aug", nprii_aug_);
            nprii_rho_core_   = section.value("nprii_rho_core", nprii_rho_core_);
            always_update_wf_ = section.value("always_update_wf", always_update_wf_);
            mixer_rms_min_    = section.value("mixer_rms_min", mixer_rms_min_);
            itsol_tol_min_    = section.value("itsol_tol_min", itsol_tol_min_);
            auto_enu_tol_     = section.value("auto_enu_tol", auto_enu_tol_);
            radial_grid_      = section.value("radial_grid", radial_grid_);
            fft_grid_size_    = section.value("fft_grid_size", fft_grid_size_);
            itsol_tol_ratio_  = section.value("itsol_tol_ratio", itsol_tol_ratio_);
            itsol_tol_scale_  = section.value("itsol_tol_scale", itsol_tol_scale_);
            sht_coverage_     = section.value("sht_coverage", sht_coverage_);
            min_occupancy_    = section.value("min_occupancy", min_occupancy_);
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
    bool hubbard_U_plus_V_{false};
    int projection_method_{0};
    struct hubbard_orbital_
    {
        int l{-1};
        int n{-1};
        std::string level;
        std::vector<double> coeff_;
        double occupancy_{0};
    };

    std::string wave_function_file_;
    std::vector<std::pair<std::string, struct hubbard_orbital_>> species;

    bool hubbard_correction() const
    {
        return hubbard_correction_;
    }

    void read(json const& parser)
    {
        if (!parser.count("hubbard")) {
            return;
        }

        if (parser["hubbard"].count("orthogonalize_hubbard_wave_functions")) {
            orthogonalize_hubbard_orbitals_ =
                parser["hubbard"].value("orthogonalize_hubbard_wave_functions", orthogonalize_hubbard_orbitals_);
        }

        if (parser["hubbard"].count("normalize_hubbard_wave_functions")) {
            normalize_hubbard_orbitals_ =
                parser["hubbard"].value("normalize_hubbard_wave_functions", normalize_hubbard_orbitals_);
        }

        if (parser["hubbard"].count("simplified_hubbard_correction")) {
            simplified_hubbard_correction_ =
                parser["hubbard"].value("simplified_hubbard_correction", simplified_hubbard_correction_);
        }
        std::vector<std::string> labels_;
        species.clear();
        labels_.clear();

        for (auto& label : parser["unit_cell"]["atom_types"]) {
            if (std::find(std::begin(labels_), std::end(labels_), label) != std::end(labels_)) {
                throw std::runtime_error("duplicate atom type label");
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
                    this->projection_method_  = 1;
                } else {
                    throw std::runtime_error(
                        "The hubbard projection method 'file' requires the option 'wave_function_file' to be defined");
                }
            }

            if (projection_method__ == "pseudo") {
                this->projection_method_ = 2;
            }
        }

        if (parser["hubbard"].count("hubbard_u_plus_v")) {
            hubbard_U_plus_V_ = true;
        }

        for (auto& label : labels_) {

            if (!parser["hubbard"].count(label)) {
                continue;
            }

            struct hubbard_orbital_ coef__;

            coef__.coeff_.clear();
            coef__.coeff_.resize(6, 0.0);

            if (parser["hubbard"][label].count("U")) {
                coef__.coeff_[0]    = parser["hubbard"][label]["U"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("J")) {
                coef__.coeff_[1]    = parser["hubbard"][label]["J"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("B")) {
                coef__.coeff_[2]    = parser["hubbard"][label]["B"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("E2")) {
                coef__.coeff_[2]    = parser["hubbard"][label]["E2"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("E3")) {
                coef__.coeff_[3]    = parser["hubbard"][label]["E3"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("alpha")) {
                coef__.coeff_[4]    = parser["hubbard"][label]["alpha"].get<double>();
                hubbard_correction_ = true;
            }

            if (parser["hubbard"][label].count("beta")) {
                coef__.coeff_[5]    = parser["hubbard"][label]["beta"].get<double>();
                hubbard_correction_ = true;
            }

            // now convert eV in Ha
            for (int s = 0; s < static_cast<int>(coef__.coeff_.size()); s++) {
                coef__.coeff_[s] /= ha2ev;
            }

            if (parser["hubbard"][label].count("l") && parser["hubbard"][label].count("n")) {
                coef__.l = parser["hubbard"][label]["l"].get<int>();
                coef__.n = parser["hubbard"][label]["n"].get<int>();
            } else {
                if (parser["hubbard"][label].count("hubbard_orbital")) {
                    coef__.level = parser["hubbard"][label]["hubbard_orbital"].get<std::string>();
                } else {
                    if (hubbard_correction_) {
                        throw std::runtime_error(
                            "you selected the hubbard correction for this atom but did not specify the atomic level");
                    }
                }
            }

            if (parser["hubbard"][label].count("occupancy")) {
                coef__.occupancy_ = parser["hubbard"][label]["occupancy"].get<double>();
            } else {
                throw std::runtime_error(
                    "This atom has hubbard correction but the occupancy is not set up. Please check your input file");
            }

            if (hubbard_correction_) {
                species.push_back(std::make_pair(label, coef__));
            }
        }

        if (!hubbard_correction_) {
            throw std::runtime_error("The hubbard section is empty");
        }
    }
};

}; // namespace sirius

#endif // __INPUT_HPP__

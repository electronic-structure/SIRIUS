class config_t
{
  public:
    nlohmann::json const& dict() const
    {
        return dict_;
    }
    /// Parameters of the mixer
    class mixer_t
    {
      public:
        mixer_t(nlohmann::json& dict__)
            : dict_(dict__)
        {
        }
        /// Type of the mixer.
        inline auto type() const
        {
            return dict_["/mixer/type"_json_pointer].get<std::string>();
        }
        inline void type(std::string type__)
        {
            dict_["/mixer/type"_json_pointer] = type__;
        }
        /// Mixing parameter
        inline auto beta() const
        {
            return dict_["/mixer/beta"_json_pointer].get<double>();
        }
        inline void beta(double beta__)
        {
            dict_["/mixer/beta"_json_pointer] = beta__;
        }
        /// Mixing ratio in case of initial linear mixing
        inline auto beta0() const
        {
            return dict_["/mixer/beta0"_json_pointer].get<double>();
        }
        inline void beta0(double beta0__)
        {
            dict_["/mixer/beta0"_json_pointer] = beta0__;
        }
        /// RMS tolerance above which the linear mixing is triggered
        inline auto linear_mix_rms_tol() const
        {
            return dict_["/mixer/linear_mix_rms_tol"_json_pointer].get<double>();
        }
        inline void linear_mix_rms_tol(double linear_mix_rms_tol__)
        {
            dict_["/mixer/linear_mix_rms_tol"_json_pointer] = linear_mix_rms_tol__;
        }
        /// Number of history steps for Broyden-type mixers
        inline auto max_history() const
        {
            return dict_["/mixer/max_history"_json_pointer].get<int>();
        }
        inline void max_history(int max_history__)
        {
            dict_["/mixer/max_history"_json_pointer] = max_history__;
        }
        /// Scaling factor for mixing parameter
        inline auto beta_scaling_factor() const
        {
            return dict_["/mixer/beta_scaling_factor"_json_pointer].get<double>();
        }
        inline void beta_scaling_factor(double beta_scaling_factor__)
        {
            dict_["/mixer/beta_scaling_factor"_json_pointer] = beta_scaling_factor__;
        }
        /// Use Hartree potential in the inner() product for residuals
        inline auto use_hartree() const
        {
            return dict_["/mixer/use_hartree"_json_pointer].get<bool>();
        }
        inline void use_hartree(bool use_hartree__)
        {
            dict_["/mixer/use_hartree"_json_pointer] = use_hartree__;
        }
      private:
        nlohmann::json& dict_;
    };
    inline auto const& mixer() const {return mixer_;}
    inline auto& mixer() {return mixer_;}
    /// Settings control the internal parameters related to the numerical implementation.
    /**
        Changing of setting parameters will have a small impact on the final result.
    */
    class settings_t
    {
      public:
        settings_t(nlohmann::json& dict__)
            : dict_(dict__)
        {
        }
        /// Point density (in a.u.^-1) for interpolating radial integrals of the local part of pseudopotential
        inline auto nprii_vloc() const
        {
            return dict_["/settings/nprii_vloc"_json_pointer].get<int>();
        }
        inline void nprii_vloc(int nprii_vloc__)
        {
            dict_["/settings/nprii_vloc"_json_pointer] = nprii_vloc__;
        }
        /// Point density (in a.u.^-1) for interpolating radial integrals of the beta projectors
        inline auto nprii_beta() const
        {
            return dict_["/settings/nprii_beta"_json_pointer].get<int>();
        }
        inline void nprii_beta(int nprii_beta__)
        {
            dict_["/settings/nprii_beta"_json_pointer] = nprii_beta__;
        }
        /// Point density (in a.u.^-1) for interpolating radial integrals of the augmentation operator
        inline auto nprii_aug() const
        {
            return dict_["/settings/nprii_aug"_json_pointer].get<int>();
        }
        inline void nprii_aug(int nprii_aug__)
        {
            dict_["/settings/nprii_aug"_json_pointer] = nprii_aug__;
        }
        /// Point density (in a.u.^-1) for interpolating radial integrals of the core charge density
        inline auto nprii_rho_core() const
        {
            return dict_["/settings/nprii_rho_core"_json_pointer].get<int>();
        }
        inline void nprii_rho_core(int nprii_rho_core__)
        {
            dict_["/settings/nprii_rho_core"_json_pointer] = nprii_rho_core__;
        }
        /// Update wave-functions in the Davdison solver even if they immediately satisfy the convergence criterion
        inline auto always_update_wf() const
        {
            return dict_["/settings/always_update_wf"_json_pointer].get<bool>();
        }
        inline void always_update_wf(bool always_update_wf__)
        {
            dict_["/settings/always_update_wf"_json_pointer] = always_update_wf__;
        }
        /// Minimum value of allowed RMS for the mixer.
        /**
            Mixer will not mix functions if the RMS between previous and current functions is below this tolerance.
        */
        inline auto mixer_rms_min() const
        {
            return dict_["/settings/mixer_rms_min"_json_pointer].get<double>();
        }
        inline void mixer_rms_min(double mixer_rms_min__)
        {
            dict_["/settings/mixer_rms_min"_json_pointer] = mixer_rms_min__;
        }
        /// Minimum tolerance of the iterative solver.
        inline auto itsol_tol_min() const
        {
            return dict_["/settings/itsol_tol_min"_json_pointer].get<double>();
        }
        inline void itsol_tol_min(double itsol_tol_min__)
        {
            dict_["/settings/itsol_tol_min"_json_pointer] = itsol_tol_min__;
        }
        /// Minimum occupancy below which the band is treated as being 'empty'
        inline auto min_occupancy() const
        {
            return dict_["/settings/min_occupancy"_json_pointer].get<double>();
        }
        inline void min_occupancy(double min_occupancy__)
        {
            dict_["/settings/min_occupancy"_json_pointer] = min_occupancy__;
        }
        /// Fine control of the empty states tolerance.
        /**
            This is the ratio between the tolerance of empty and occupied states. Used in the code like this:
            \code{.cpp}
            // tolerance of occupied bands
            double tol = ctx_.iterative_solver_tolerance();
            // final tolerance of empty bands
            double empy_tol = std::max(tol * ctx_.settings().itsol_tol_ratio_, itso.empty_states_tolerance_);
            \endcode
        */
        inline auto itsol_tol_ratio() const
        {
            return dict_["/settings/itsol_tol_ratio"_json_pointer].get<double>();
        }
        inline void itsol_tol_ratio(double itsol_tol_ratio__)
        {
            dict_["/settings/itsol_tol_ratio"_json_pointer] = itsol_tol_ratio__;
        }
        /// Scaling parameters of the iterative  solver tolerance.
        /**
            First number is the scaling of density RMS, that gives the estimate of the new 
            tolerance. Second number is the scaling of the old tolerance. New tolerance is then the minimum 
            between the two. This is how it is done in the code: 
            \code{.cpp}
            double old_tol = ctx_.iterative_solver_tolerance();
            // estimate new tolerance of iterative solver
            double tol = std::min(ctx_.settings().itsol_tol_scale_[0] * rms, ctx_.settings().itsol_tol_scale_[1] * old_tol);
            tol = std::max(ctx_.settings().itsol_tol_min_, tol);
            // set new tolerance of iterative solver
            ctx_.iterative_solver_tolerance(tol);\endcode
        */
        inline auto itsol_tol_scale() const
        {
            return dict_["/settings/itsol_tol_scale"_json_pointer].get<std::array<double, 2>>();
        }
        inline void itsol_tol_scale(std::array<double, 2> itsol_tol_scale__)
        {
            dict_["/settings/itsol_tol_scale"_json_pointer] = itsol_tol_scale__;
        }
        /// Tolerance to recompute the LAPW linearisation energies.
        inline auto auto_enu_tol() const
        {
            return dict_["/settings/auto_enu_tol"_json_pointer].get<double>();
        }
        inline void auto_enu_tol(double auto_enu_tol__)
        {
            dict_["/settings/auto_enu_tol"_json_pointer] = auto_enu_tol__;
        }
        /// Initial dimenstions for the fine-grain FFT grid
        inline auto fft_grid_size() const
        {
            return dict_["/settings/fft_grid_size"_json_pointer].get<std::array<int, 3>>();
        }
        inline void fft_grid_size(std::array<int, 3> fft_grid_size__)
        {
            dict_["/settings/fft_grid_size"_json_pointer] = fft_grid_size__;
        }
        /// Default radial grid for LAPW species.
        inline auto radial_grid() const
        {
            return dict_["/settings/radial_grid"_json_pointer].get<std::string>();
        }
        inline void radial_grid(std::string radial_grid__)
        {
            dict_["/settings/radial_grid"_json_pointer] = radial_grid__;
        }
        /// Coverage of sphere in case of spherical harmonics transformation
        /**
            0 is Lebedev-Laikov coverage, 1 is unifrom coverage
        */
        inline auto sht_coverage() const
        {
            return dict_["/settings/sht_coverage"_json_pointer].get<int>();
        }
        inline void sht_coverage(int sht_coverage__)
        {
            dict_["/settings/sht_coverage"_json_pointer] = sht_coverage__;
        }
      private:
        nlohmann::json& dict_;
    };
    inline auto const& settings() const {return settings_;}
    inline auto& settings() {return settings_;}
    /// Unit cell representation
    class unit_cell_t
    {
      public:
        unit_cell_t(nlohmann::json& dict__)
            : dict_(dict__)
        {
        }
        /// Three non-collinear vectors of the primitive unit cell.
        inline auto lattice_vectors() const
        {
            return dict_["/unit_cell/lattice_vectors"_json_pointer].get<std::array<std::array<double, 3>, 3>>();
        }
        inline void lattice_vectors(std::array<std::array<double, 3>, 3> lattice_vectors__)
        {
            dict_["/unit_cell/lattice_vectors"_json_pointer] = lattice_vectors__;
        }
        /// Scaling factor for the lattice vectors
        /**
            Lattice vectors are multiplied by this constant.
        */
        inline auto lattice_vectors_scale() const
        {
            return dict_["/unit_cell/lattice_vectors_scale"_json_pointer].get<double>();
        }
        inline void lattice_vectors_scale(double lattice_vectors_scale__)
        {
            dict_["/unit_cell/lattice_vectors_scale"_json_pointer] = lattice_vectors_scale__;
        }
        /// Type of atomic coordinates: lattice, atomic units or Angstroms
        inline auto atom_coordinate_units() const
        {
            return dict_["/unit_cell/atom_coordinate_units"_json_pointer].get<std::string>();
        }
        inline void atom_coordinate_units(std::string atom_coordinate_units__)
        {
            dict_["/unit_cell/atom_coordinate_units"_json_pointer] = atom_coordinate_units__;
        }
        inline auto atom_types() const
        {
            return dict_["/unit_cell/atom_types"_json_pointer].get<std::vector<std::string>>();
        }
        inline void atom_types(std::vector<std::string> atom_types__)
        {
            dict_["/unit_cell/atom_types"_json_pointer] = atom_types__;
        }
        /// Mapping between atom type labels and atomic files
        inline auto atom_files(std::string label__) const
        {
            nlohmann::json::json_pointer p("/unit_cell/atom_files");
            return dict_[p / label__].get<std::string>();
        }
        /// Atomic coordinates
        inline auto atoms(std::string label__) const
        {
            nlohmann::json::json_pointer p("/unit_cell/atoms");
            return dict_[p / label__].get<std::vector<std::vector<double>>>();
        }
      private:
        nlohmann::json& dict_;
    };
    inline auto const& unit_cell() const {return unit_cell_;}
    inline auto& unit_cell() {return unit_cell_;}
    /// Parameters of the iterative solver.
    class iterative_solver_t
    {
      public:
        iterative_solver_t(nlohmann::json& dict__)
            : dict_(dict__)
        {
        }
        /// Type of the iterative solver.
        inline auto type() const
        {
            return dict_["/iterative_solver/type"_json_pointer].get<std::string>();
        }
        inline void type(std::string type__)
        {
            dict_["/iterative_solver/type"_json_pointer] = type__;
        }
        /// Number of steps (iterations) of the solver.
        inline auto num_steps() const
        {
            return dict_["/iterative_solver/num_steps"_json_pointer].get<int>();
        }
        inline void num_steps(int num_steps__)
        {
            dict_["/iterative_solver/num_steps"_json_pointer] = num_steps__;
        }
        /// Size of the variational subspace is this number times the number of bands.
        inline auto subspace_size() const
        {
            return dict_["/iterative_solver/subspace_size"_json_pointer].get<int>();
        }
        inline void subspace_size(int subspace_size__)
        {
            dict_["/iterative_solver/subspace_size"_json_pointer] = subspace_size__;
        }
        /// Lock eigenvectors of the smallest eigenvalues when they have converged at restart.
        inline auto locking() const
        {
            return dict_["/iterative_solver/locking"_json_pointer].get<bool>();
        }
        inline void locking(bool locking__)
        {
            dict_["/iterative_solver/locking"_json_pointer] = locking__;
        }
        /// Restart early when the ratio unconverged vs lockable vectors drops below this threshold.
        /**
            When there's just a few vectors left unconverged, it can be more efficient to lock the converged ones,
            such that the dense eigenproblem solved in each Davidson iteration has lower dimension.
            Restarting has some overhead in that it requires updating wave functions.
        */
        inline auto early_restart() const
        {
            return dict_["/iterative_solver/early_restart"_json_pointer].get<double>();
        }
        inline void early_restart(double early_restart__)
        {
            dict_["/iterative_solver/early_restart"_json_pointer] = early_restart__;
        }
        /// Tolerance for the eigen-energy difference \f$ |\epsilon_i^{old} - \epsilon_i^{new} | \f$
        /**
            This parameter is reduced during the SCF cycle to reach the high accuracy of the wave-functions.
        */
        inline auto energy_tolerance() const
        {
            return dict_["/iterative_solver/energy_tolerance"_json_pointer].get<double>();
        }
        inline void energy_tolerance(double energy_tolerance__)
        {
            dict_["/iterative_solver/energy_tolerance"_json_pointer] = energy_tolerance__;
        }
        /// Tolerance for the residual L2 norm.
        inline auto residual_tolerance() const
        {
            return dict_["/iterative_solver/residual_tolerance"_json_pointer].get<double>();
        }
        inline void residual_tolerance(double residual_tolerance__)
        {
            dict_["/iterative_solver/residual_tolerance"_json_pointer] = residual_tolerance__;
        }
        /// Relative tolerance for the residual L2 norm. (0 means this criterion is effectively not used.
        inline auto relative_tolerance() const
        {
            return dict_["/iterative_solver/relative_tolerance"_json_pointer].get<double>();
        }
        inline void relative_tolerance(double relative_tolerance__)
        {
            dict_["/iterative_solver/relative_tolerance"_json_pointer] = relative_tolerance__;
        }
        /// Additional tolerance for empty states.
        /**
            Setting this variable to 0 will treat empty states with the same tolerance as occupied states.
        */
        inline auto empty_states_tolerance() const
        {
            return dict_["/iterative_solver/empty_states_tolerance"_json_pointer].get<double>();
        }
        inline void empty_states_tolerance(double empty_states_tolerance__)
        {
            dict_["/iterative_solver/empty_states_tolerance"_json_pointer] = empty_states_tolerance__;
        }
        /// Defines the flavour of the iterative solver.
        /**
            If converge_by_energy is set to 0, then the residuals are estimated by their norm. If converge_by_energy
            is set to 1 then the residuals are estimated by the eigen-energy difference. This allows to estimate the
            unconverged residuals and then compute only the unconverged ones.
        */
        inline auto converge_by_energy() const
        {
            return dict_["/iterative_solver/converge_by_energy"_json_pointer].get<int>();
        }
        inline void converge_by_energy(int converge_by_energy__)
        {
            dict_["/iterative_solver/converge_by_energy"_json_pointer] = converge_by_energy__;
        }
        /// Minimum number of residuals to continue iterative diagonalization process.
        inline auto min_num_res() const
        {
            return dict_["/iterative_solver/min_num_res"_json_pointer].get<int>();
        }
        inline void min_num_res(int min_num_res__)
        {
            dict_["/iterative_solver/min_num_res"_json_pointer] = min_num_res__;
        }
        /// Number of singular components for the LAPW Davidson solver.
        /**
            Singular components are the eigen-vectors of the APW-APW block of overlap matrix
        */
        inline auto num_singular() const
        {
            return dict_["/iterative_solver/num_singular"_json_pointer].get<int>();
        }
        inline void num_singular(int num_singular__)
        {
            dict_["/iterative_solver/num_singular"_json_pointer] = num_singular__;
        }
        /// Initialize eigen-values with previous (old) values.
        inline auto init_eval_old() const
        {
            return dict_["/iterative_solver/init_eval_old"_json_pointer].get<bool>();
        }
        inline void init_eval_old(bool init_eval_old__)
        {
            dict_["/iterative_solver/init_eval_old"_json_pointer] = init_eval_old__;
        }
        /// Tell how to initialize the subspace.
        /**
            It can be either 'lcao', i.e. start from the linear combination of atomic orbitals or
            'random' – start from the randomized wave functions.
        */
        inline auto init_subspace() const
        {
            return dict_["/iterative_solver/init_subspace"_json_pointer].get<std::string>();
        }
        inline void init_subspace(std::string init_subspace__)
        {
            dict_["/iterative_solver/init_subspace"_json_pointer] = init_subspace__;
        }
      private:
        nlohmann::json& dict_;
    };
    inline auto const& iterative_solver() const {return iterative_solver_;}
    inline auto& iterative_solver() {return iterative_solver_;}
  private:
    mixer_t mixer_{dict_};
    settings_t settings_{dict_};
    unit_cell_t unit_cell_{dict_};
    iterative_solver_t iterative_solver_{dict_};
  protected:
    nlohmann::json dict_;
};


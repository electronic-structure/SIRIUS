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
  private:
    mixer_t mixer_{dict_};
    settings_t settings_{dict_};
    unit_cell_t unit_cell_{dict_};
  protected:
    nlohmann::json dict_;
};


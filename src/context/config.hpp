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
  private:
    mixer_t mixer_{dict_};
  protected:
    nlohmann::json dict_;
};


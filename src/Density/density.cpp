#include "density.h"

namespace sirius {

Density::Density(Simulation_context& ctx__)
    : ctx_(ctx__),
      parameters_(ctx__.parameters()),
      unit_cell_(ctx_.unit_cell()),
      fft_(ctx_.fft()),
      rho_pseudo_core_(nullptr),
      gaunt_coefs_(nullptr),
      high_freq_mixer_(nullptr),
      low_freq_mixer_(nullptr),
      mixer_(nullptr)
{
    rho_ = new Periodic_function<double>(ctx_, parameters_.lmmax_rho());

    /* core density of the pseudopotential method */
    if (!parameters_.full_potential())
    {
        rho_pseudo_core_ = new Periodic_function<double>(ctx_, 0, false);
        rho_pseudo_core_->allocate(false, true);
        rho_pseudo_core_->zero();

        generate_pseudo_core_charge_density();
    }

    for (int i = 0; i < parameters_.num_mag_dims(); i++)
    {
        magnetization_[i] = new Periodic_function<double>(ctx_, parameters_.lmmax_rho());
    }
    
    /* never change this order!!! */
    dmat_spins_.clear();
    dmat_spins_.push_back(std::pair<int, int>(0, 0));
    dmat_spins_.push_back(std::pair<int, int>(1, 1));
    dmat_spins_.push_back(std::pair<int, int>(0, 1));
    
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_apw(), parameters_.lmax_rho(), 
                                                                  parameters_.lmax_apw(), SHT::gaunt_hybrid);
            break;
        }
        case full_potential_pwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<double_complex>(parameters_.lmax_pw(), parameters_.lmax_rho(), 
                                                                  parameters_.lmax_pw(), SHT::gaunt_hybrid);
            break;
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            break;
        }
    }

    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_rho());

    if (!parameters_.full_potential())
    {
        high_freq_mixer_ = new Linear_mixer<double_complex>((ctx_.fft()->num_gvec() - ctx_.fft_coarse()->num_gvec()),
                                                            parameters_.mixer_input_section().beta_, ctx_.comm());

        std::vector<double> weights(ctx_.fft_coarse()->num_gvec());
        weights[0] = 0;
        for (int ig = 1; ig < ctx_.fft_coarse()->num_gvec(); ig++)
            weights[ig] = fourpi * unit_cell_.omega() / std::pow(ctx_.fft_coarse()->gvec_len(ig), 2);

        if (parameters_.mixer_input_section().type_ == "linear")
        {
            low_freq_mixer_ = new Linear_mixer<double_complex>(ctx_.fft_coarse()->num_gvec(),
                                                               parameters_.mixer_input_section().beta_,
                                                               ctx_.comm());
        }
        else if (parameters_.mixer_input_section().type_ == "broyden2")
        {
            low_freq_mixer_ = new Broyden_mixer<double_complex>(ctx_.fft_coarse()->num_gvec(),
                                                                parameters_.mixer_input_section().max_history_,
                                                                parameters_.mixer_input_section().beta_,
                                                                weights,
                                                                ctx_.comm());
        } 
        else if (parameters_.mixer_input_section().type_ == "broyden1")
        {

            low_freq_mixer_ = new Broyden_modified_mixer<double_complex>(ctx_.fft_coarse()->num_gvec(),
                                                                         parameters_.mixer_input_section().max_history_,
                                                                         parameters_.mixer_input_section().beta_,
                                                                         weights,
                                                                         ctx_.comm());
        }
        else
        {
            TERMINATE("wrong mixer type");
        }
    }

    if (parameters_.full_potential())
    {
        if (parameters_.mixer_input_section().type_ == "linear")
        {
            mixer_ = new Linear_mixer<double>(size(),
                                              parameters_.mixer_input_section().beta_,
                                              ctx_.comm());
        }
        else if (parameters_.mixer_input_section().type_ == "broyden2")
        {
            std::vector<double> weights;
            mixer_ = new Broyden_mixer<double>(size(),
                                               parameters_.mixer_input_section().max_history_,
                                               parameters_.mixer_input_section().beta_,
                                               weights,
                                               ctx_.comm());

        }
        else if (parameters_.mixer_input_section().type_ == "broyden1")
        {
            std::vector<double> weights;
            mixer_ = new Broyden_modified_mixer<double>(size(),
                                                        parameters_.mixer_input_section().max_history_,
                                                        parameters_.mixer_input_section().beta_,
                                                        weights,
                                                        ctx_.comm());
        }
        else
        {
            TERMINATE("wrong mixer type");
        }
    }

    //splindex<block> spl_num_gvec(fft_->num_gvec(), ctx_.comm().size(), ctx_.comm().rank());
    
    //== gvec_phase_factors_ = mdarray<double_complex, 2>(spl_num_gvec.local_size(), unit_cell_.num_atoms());
    //== #pragma omp parallel for
    //== for (int igloc = 0; igloc < (int)spl_num_gvec.local_size(); igloc++)
    //== {
    //==     int ig = (int)spl_num_gvec[igloc];
    //==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) gvec_phase_factors_(igloc, ia) = ctx_.reciprocal_lattice()->gvec_phase_factor(ig, ia);
    //== }
}

Density::~Density()
{
    delete rho_;
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete magnetization_[j];

    if (rho_pseudo_core_ != nullptr) delete rho_pseudo_core_;
    if (gaunt_coefs_ != nullptr) delete gaunt_coefs_;
    if (low_freq_mixer_ != nullptr) delete low_freq_mixer_;
    if (high_freq_mixer_ != nullptr) delete high_freq_mixer_;
    if (mixer_ != nullptr) delete mixer_;
}

};

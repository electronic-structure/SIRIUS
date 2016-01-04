#include "density.h"

namespace sirius {

Density::Density(Simulation_context& ctx__)
    : ctx_(ctx__),
      parameters_(ctx__.parameters()),
      unit_cell_(ctx_.unit_cell()),
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
        rho_pseudo_core_->allocate(false);
        rho_pseudo_core_->zero();

        generate_pseudo_core_charge_density();
    }

    for (int i = 0; i < parameters_.num_mag_dims(); i++)
    {
        magnetization_[i] = new Periodic_function<double>(ctx_, parameters_.lmmax_rho());
    }
    
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
        lf_gvec_ = std::vector<int>(ctx_.gvec_coarse().num_gvec());
        std::vector<double> weights(ctx_.gvec_coarse().num_gvec() * (1 + parameters_.num_mag_dims()), 1.0);

        weights[0] = 0;
        lf_gvec_[0] = 0;

        for (int ig = 1; ig < ctx_.gvec_coarse().num_gvec(); ig++)
        {
            auto G = ctx_.gvec_coarse()[ig];
            /* save index of low-frequency G-vector */
            lf_gvec_[ig] = ctx_.gvec().index_by_gvec(G);
            weights[ig] = fourpi * unit_cell_.omega() / std::pow(ctx_.gvec_coarse().gvec_len(ig), 2);
        }

        /* find high-frequency G-vectors */
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++)
        {
            if (ctx_.gvec().gvec_len(ig) > 2 * parameters_.gk_cutoff())
                hf_gvec_.push_back(ig);
        }

        if (static_cast<int>(hf_gvec_.size()) != ctx_.gvec().num_gvec() - ctx_.gvec_coarse().num_gvec())
        {
            std::stringstream s;
            s << "Wrong count of high-frequency G-vectors" << std::endl
              << "number of found high-frequency G-vectors: " << hf_gvec_.size() << std::endl
              << "expectd number of high-frequency G-vectors: " << ctx_.gvec().num_gvec() - ctx_.gvec_coarse().num_gvec() << std::endl
              << "G-vector cutoff: " <<  parameters_.gk_cutoff();
            TERMINATE(s);
        }

        high_freq_mixer_ = new Linear_mixer<double_complex>(hf_gvec_.size() * (1 + parameters_.num_mag_dims()),
                                                            parameters_.mixer_input_section().beta_, ctx_.comm());

        if (parameters_.mixer_input_section().type_ == "linear")
        {
            low_freq_mixer_ = new Linear_mixer<double_complex>(lf_gvec_.size() * (1 + parameters_.num_mag_dims()),
                                                               parameters_.mixer_input_section().beta_, ctx_.comm());
        }
        else if (parameters_.mixer_input_section().type_ == "broyden1")
        {

            low_freq_mixer_ = new Broyden1<double_complex>(lf_gvec_.size() * (1 + parameters_.num_mag_dims()),
                                                           parameters_.mixer_input_section().max_history_,
                                                           parameters_.mixer_input_section().beta_,
                                                           weights,
                                                           ctx_.comm());
        }
        else if (parameters_.mixer_input_section().type_ == "broyden2")
        {
            low_freq_mixer_ = new Broyden2<double_complex>(lf_gvec_.size() * (1 + parameters_.num_mag_dims()),
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
        else if (parameters_.mixer_input_section().type_ == "broyden1")
        {
            std::vector<double> weights;
            mixer_ = new Broyden1<double>(size(),
                                          parameters_.mixer_input_section().max_history_,
                                          parameters_.mixer_input_section().beta_,
                                          weights,
                                          ctx_.comm());
        }
        else if (parameters_.mixer_input_section().type_ == "broyden2")
        {
            std::vector<double> weights;
            mixer_ = new Broyden2<double>(size(),
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

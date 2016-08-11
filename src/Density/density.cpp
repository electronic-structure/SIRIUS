#include "density.h"

namespace sirius {

Density::Density(Simulation_context& ctx__)
    : ctx_(ctx__),
      unit_cell_(ctx_.unit_cell()),
      rho_pseudo_core_(nullptr),
      gaunt_coefs_(nullptr),
      high_freq_mixer_(nullptr),
      low_freq_mixer_(nullptr),
      mixer_(nullptr)
{
    rho_ = new Periodic_function<double>(ctx_, ctx_.lmmax_rho(), 1);

    /* core density of the pseudopotential method */
    if (!ctx_.full_potential())
    {
        rho_pseudo_core_ = new Periodic_function<double>(ctx_, 0, false);
        rho_pseudo_core_->zero();

        generate_pseudo_core_charge_density();
    }

    for (int i = 0; i < ctx_.num_mag_dims(); i++)
        magnetization_[i] = new Periodic_function<double>(ctx_, ctx_.lmmax_rho(), 1);
    
    using gc_z = Gaunt_coefficients<double_complex>;

    switch (ctx_.esm_type())
    {
        case full_potential_lapwlo:
        {
            gaunt_coefs_ = std::unique_ptr<gc_z>(new gc_z(ctx_.lmax_apw(), ctx_.lmax_rho(), ctx_.lmax_apw(), SHT::gaunt_hybrid));
            break;
        }
        case full_potential_pwlo:
        {
            gaunt_coefs_ = std::unique_ptr<gc_z>(new gc_z(ctx_.lmax_pw(), ctx_.lmax_rho(), ctx_.lmax_pw(), SHT::gaunt_hybrid));
            break;
        }

        case paw_pseudopotential:
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            break;
        }
    }

    l_by_lm_ = Utils::l_by_lm(ctx_.lmax_rho());

    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric. */
    int ndm = std::max(ctx_.num_mag_dims(), ctx_.num_spins());

    if (ctx_.full_potential()) {
        density_matrix_ = mdarray<double_complex, 4>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                                     ndm, unit_cell_.spl_num_atoms().local_size());
    } else {
        density_matrix_ = mdarray<double_complex, 4>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                                     ndm, unit_cell_.num_atoms());
    }

    if (!ctx_.full_potential())
    {
        lf_gvec_ = std::vector<int>(ctx_.gvec_coarse().num_gvec());
        std::vector<double> weights(ctx_.gvec_coarse().num_gvec() * (1 + ctx_.num_mag_dims()), 1.0);
        for (size_t i = 0; i < density_matrix_.size(); i++) {
            weights.push_back(0);
        }

        for(int i= weights.size() - 1; i >= weights.size() - density_matrix_.size() ; i--)
        {
            weights[i] = 0.0;
        }

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
            if (ctx_.gvec().gvec_len(ig) > 2 * ctx_.gk_cutoff()) hf_gvec_.push_back(ig);
        }

        if (static_cast<int>(hf_gvec_.size()) != ctx_.gvec().num_gvec() - ctx_.gvec_coarse().num_gvec())
        {
            std::stringstream s;
            s << "Wrong count of high-frequency G-vectors" << std::endl
              << "number of found high-frequency G-vectors: " << hf_gvec_.size() << std::endl
              << "expected number of high-frequency G-vectors: " << ctx_.gvec().num_gvec() - ctx_.gvec_coarse().num_gvec() << std::endl
              << "G-vector cutoff: " <<  ctx_.gk_cutoff();
            TERMINATE(s);
        }

        high_freq_mixer_ = new Linear_mixer<double_complex>(hf_gvec_.size() * (1 + ctx_.num_mag_dims()),
                                                            ctx_.mixer_input_section().beta_, ctx_.comm());

        if (ctx_.mixer_input_section().type_ == "linear")
        {
            low_freq_mixer_ = new Linear_mixer<double_complex>(lf_gvec_.size() * (1 + ctx_.num_mag_dims()),
                                                               ctx_.mixer_input_section().beta_, ctx_.comm());
        }
        else if (ctx_.mixer_input_section().type_ == "broyden1")
        {

            low_freq_mixer_ = new Broyden1<double_complex>(lf_gvec_.size() * (1 + ctx_.num_mag_dims()) + density_matrix_.size(),
                                                           ctx_.mixer_input_section().max_history_,
                                                           ctx_.mixer_input_section().beta_,
                                                           weights,
                                                           ctx_.comm());
        }
        else if (ctx_.mixer_input_section().type_ == "broyden2")
        {
            low_freq_mixer_ = new Broyden2<double_complex>(lf_gvec_.size() * (1 + ctx_.num_mag_dims()),
                                                           ctx_.mixer_input_section().max_history_,
                                                           ctx_.mixer_input_section().beta_,
                                                           ctx_.mixer_input_section().beta0_,
                                                           ctx_.mixer_input_section().linear_mix_rms_tol_,
                                                           weights,
                                                           ctx_.comm());
        } 
        else
        {
            TERMINATE("wrong mixer type");
        }
    }

    if (ctx_.full_potential())
    {
        if (ctx_.mixer_input_section().type_ == "linear")
        {
            mixer_ = new Linear_mixer<double>(size(),
                                              ctx_.mixer_input_section().beta_,
                                              ctx_.comm());
        }
        else if (ctx_.mixer_input_section().type_ == "broyden1")
        {
            std::vector<double> weights;
            mixer_ = new Broyden1<double>(size(),
                                          ctx_.mixer_input_section().max_history_,
                                          ctx_.mixer_input_section().beta_,
                                          weights,
                                          ctx_.comm());
        }
        else if (ctx_.mixer_input_section().type_ == "broyden2")
        {
            std::vector<double> weights;
            mixer_ = new Broyden2<double>(size(),
                                          ctx_.mixer_input_section().max_history_,
                                          ctx_.mixer_input_section().beta_,
                                          ctx_.mixer_input_section().beta0_,
                                          ctx_.mixer_input_section().linear_mix_rms_tol_,
                                          weights,
                                          ctx_.comm());

        }
        else
        {
            TERMINATE("wrong mixer type");
        }
    }


    std::cout<<"SPL ATOMS "<< unit_cell_.spl_num_atoms().local_size()<< std::endl;

    //--- Allocate local PAW density arrays ---

    for(int i = 0; i < unit_cell_.spl_num_atoms().local_size(); i++)
    {
        int ia = unit_cell_.spl_num_atoms(i);

        auto& atom = unit_cell_.atom(ia);

        auto& atype = atom.type();

        int n_mt_points = atype.num_mt_points();

        int rad_func_lmax = atype.indexr().lmax_lo();
        int n_rho_lm_comp = (2 * rad_func_lmax + 1) * (2 * rad_func_lmax + 1);

        // allocate
        mdarray<double, 2> ae_atom_density(n_rho_lm_comp, n_mt_points);
        mdarray<double, 2> ps_atom_density(n_rho_lm_comp, n_mt_points);

        // add
        paw_ae_local_density_.push_back(std::move(ae_atom_density));
        paw_ps_local_density_.push_back(std::move(ps_atom_density));

        // magnetization
        mdarray<double, 3> ae_atom_magn(n_rho_lm_comp, n_mt_points, 3);
        mdarray<double, 3> ps_atom_magn(n_rho_lm_comp, n_mt_points, 3);

        ae_atom_magn.zero();
        ps_atom_magn.zero();

        paw_ae_local_magnetization_.push_back(std::move(ae_atom_magn));
        paw_ps_local_magnetization_.push_back(std::move(ps_atom_magn));

    }
    std::cout<<"paw density init done"<< std::endl;
}

Density::~Density()
{
    delete rho_;
    for (int j = 0; j < ctx_.num_mag_dims(); j++) delete magnetization_[j];

    if (rho_pseudo_core_ != nullptr) delete rho_pseudo_core_;
    if (low_freq_mixer_ != nullptr) delete low_freq_mixer_;
    if (high_freq_mixer_ != nullptr) delete high_freq_mixer_;
    if (mixer_ != nullptr) delete mixer_;
}

};

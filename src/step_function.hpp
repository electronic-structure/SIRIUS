void Step_function::init()
{
    Timer t("sirius::Step_function::init");

    auto fft = reciprocal_lattice_->fft();
    
    mdarray<double, 2> ffac((int)reciprocal_lattice_->num_gvec_shells_total(), unit_cell_->num_atom_types());
    get_step_function_form_factors(ffac);

    step_function_pw_.resize(fft->size());
    step_function_.resize(fft->size());
    
    memset(&step_function_pw_[0], 0, fft->size() * sizeof(complex16));
    
    #pragma omp parallel for default(shared)
    for (int igloc = 0; igloc < fft->local_size(); igloc++)
    {
        int ig = fft->global_index(igloc);
        int igs = reciprocal_lattice_->gvec_shell<global>(ig);

        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++)
        {            
            int iat = unit_cell_->atom(ia)->type_id();
            step_function_pw_[ig] -= conj(reciprocal_lattice_->gvec_phase_factor<global>(ig, ia)) * ffac(igs, iat);

        }
    }
    Platform::allgather(&step_function_pw_[0], fft->global_offset(), fft->local_size());
    
    step_function_pw_[0] += 1.0;

    fft->input(fft->size(), reciprocal_lattice_->fft_index(), &step_function_pw_[0]);
    fft->transform(1);
    fft->output(&step_function_[0]);
    
    double vit = 0.0;
    for (int i = 0; i < fft->size(); i++) vit += step_function_[i] * unit_cell_->omega() / fft->size();
    
    if (fabs(vit - unit_cell_->volume_it()) > 1e-10)
    {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << fabs(vit - unit_cell_->volume_it());
        warning_global(__FILE__, __LINE__, s);
    }
}

void Step_function::get_step_function_form_factors(mdarray<double, 2>& ffac)
{
    ffac.zero();
    
    double fourpi_omega = fourpi / unit_cell_->omega();

    splindex<block> spl_num_gvec_shells(ffac.size(0), Platform::num_mpi_ranks(), Platform::mpi_rank());

    #pragma omp parallel for default(shared)
    for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
    {
        int igs = spl_num_gvec_shells[igsloc];
        double g = reciprocal_lattice_->gvec_shell_len(igs);
        double g3inv = (igs) ? 1.0 / pow(g, 3) : 0.0;

        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {            
            double R = unit_cell_->atom_type(iat)->mt_radius();
            double gR = g * R;

            if (igs == 0)
            {
                ffac(igs, iat) = fourpi_omega * pow(R, 3) / 3.0;
            }
            else
            {
                ffac(igs, iat) = fourpi_omega * (sin(gR) - gR * cos(gR)) * g3inv;
            }
        }
    }

    for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++) 
        Platform::allgather(&ffac(0, iat), spl_num_gvec_shells.global_offset(), spl_num_gvec_shells.local_size());
}

void Step_function::init()
{
    Timer t("sirius::Step_function::init");
    
    mdarray<double, 2> ffac((int)gvec_shell_len_.size(), num_atom_types());
    get_step_function_form_factors(ffac);

    step_function_pw_.resize(fft().size());
    step_function_.resize(fft().size());
    
    memset(&step_function_pw_[0], 0, fft().size() * sizeof(complex16));
    
    #pragma omp parallel for default(shared)
    for (int igloc = 0; igloc < spl_fft_size().local_size(); igloc++)
    {
        int ig = spl_fft_size(igloc);
        int igs = gvec_shell<global>(ig);

        for (int ia = 0; ia < num_atoms(); ia++)
        {            
            int iat = atom_type_index_by_id(atom(ia)->type_id());
            step_function_pw_[ig] -= conj(gvec_phase_factor<global>(ig, ia)) * ffac(igs, iat);

        }
    }
    Platform::allgather(&step_function_pw_[0], spl_fft_size().global_offset(), spl_fft_size().local_size());
    
    step_function_pw_[0] += 1.0;

    fft().input(fft().size(), fft_index(), &step_function_pw_[0]);
    fft().transform(1);
    fft().output(&step_function_[0]);
    
    volume_mt_ = 0.0;
    for (int ia = 0; ia < num_atoms(); ia++) volume_mt_ += fourpi * pow(atom(ia)->type()->mt_radius(), 3) / 3.0; 
    
    volume_it_ = omega() - volume_mt_;
    double vit = 0.0;
    for (int i = 0; i < fft().size(); i++) vit += step_function_[i] * omega() / fft().size();
    
    if (fabs(vit - volume_it_) > 1e-10)
    {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << vit - volume_it_;
        warning_global(__FILE__, __LINE__, s);
    }
}

void Step_function::print_info()
{
    printf("\n");
    printf("Unit cell volume : %f\n", omega());
    printf("MT volume        : %f (%i%%)\n", volume_mt(), int(volume_mt() * 100 / omega()));
    printf("IT volume        : %f (%i%%)\n", volume_it(), int(volume_it() * 100 / omega()));
}

void Step_function::get_step_function_form_factors(mdarray<double, 2>& ffac)
{
    ffac.zero();
    
    double fourpi_omega = fourpi / omega();

    splindex<block> spl_num_gvec_shells(ffac.size(0), Platform::num_mpi_ranks(), Platform::mpi_rank());

    #pragma omp parallel for default(shared)
    for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
    {
        int igs = spl_num_gvec_shells[igsloc];
        double g = gvec_shell_len(igs);
        double g3inv = (igs) ? 1.0 / pow(g, 3) : 0.0;

        for (int iat = 0; iat < num_atom_types(); iat++)
        {            
            double R = atom_type(iat)->mt_radius();
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

    for (int iat = 0; iat < num_atom_types(); iat++) 
        Platform::allgather(&ffac(0, iat), spl_num_gvec_shells.global_offset(), spl_num_gvec_shells.local_size());
}

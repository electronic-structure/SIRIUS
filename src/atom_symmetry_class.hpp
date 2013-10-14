void Atom_symmetry_class::generate_aw_radial_functions()
{
    int nmtp = atom_type_->num_mt_points();
    double R = atom_type_->mt_radius();
   
    Radial_solver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
    
    #pragma omp parallel default(shared)
    {
        Spline<double> s(nmtp, atom_type_->radial_grid());
        
        std::vector<double> p;
        std::vector<double> hp;
        
        double dpdr[atom_type_->max_aw_order()];
        
        #pragma omp for schedule(dynamic, 1)
        for (int l = 0; l < num_aw_descriptors(); l++)
        {
            for (int order = 0; order < (int)aw_descriptor(l).size(); order++)
            {
                radial_solution_descriptor& rsd = aw_descriptor(l)[order];

                int idxrf = atom_type_->indexr().index_by_l_order(l, order);

                // find linearization energies
                switch (rsd.auto_enu)
                {
                    case 1:
                    {
                        solver.bound_state(rsd.n, rsd.l, spherical_potential_, rsd.enu, p);
                        break;
                    }
                    case 2:
                    { 
                        rsd.enu = solver.find_enu(rsd.n, rsd.l, spherical_potential_, rsd.enu);
                        break;
                    }
                }

                solver.solve_in_mt(rsd.l, rsd.enu, rsd.dme, spherical_potential_, p, hp, dpdr[order]);

                // normalize
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(p[ir], 2);
                double norm = 1.0 / sqrt(s.interpolate().integrate(0));

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf, 0) = p[ir] * norm;
                    radial_functions_(ir, idxrf, 1) = hp[ir] * norm;
                }
                dpdr[order] *= norm;

                // orthogonalize
                for (int order1 = 0; order1 < order; order1++)
                {
                    int idxrf1 = atom_type_->indexr().index_by_l_order(l, order1);

                    for (int ir = 0; ir < nmtp; ir++) 
                        s[ir] = radial_functions_(ir, idxrf, 0) * radial_functions_(ir, idxrf1, 0);

                    double t1 = s.interpolate().integrate(0);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        radial_functions_(ir, idxrf, 0) -= radial_functions_(ir, idxrf1, 0) * t1;
                        radial_functions_(ir, idxrf, 1) -= radial_functions_(ir, idxrf1, 1) * t1;
                    }
                    dpdr[order] -= t1 * dpdr[order1];
                }

                // normalize again
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(radial_functions_(ir, idxrf, 0), 2);
                norm = s.interpolate().integrate(0);

                if (fabs(norm) < 1e-10) error_local(__FILE__, __LINE__, "aw radial functions are linearly dependent");

                norm = 1.0 / sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf, 0) *= norm;
                    radial_functions_(ir, idxrf, 1) *= norm;
                }
                dpdr[order] *= norm;

                // radial derivative
                double rderiv = dpdr[order];

                aw_surface_derivatives_(order, l) = (rderiv - radial_functions_(nmtp - 1, idxrf, 0) / R) / R;
                
                if (debug_level > 1)
                {
                    printf("atom class id : %i  l : %i  order : %i  radial function value at MT : %f\n", 
                           id_, l, order, radial_functions_(nmtp - 1, idxrf, 0));
                }
            }
        }
    }
}

void Atom_symmetry_class::generate_lo_radial_functions()
{
    int nmtp = atom_type_->num_mt_points();
    double R = atom_type_->mt_radius();
    Radial_solver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
    
    #pragma omp parallel default(shared)
    {
        Spline<double> s(nmtp, atom_type_->radial_grid());
        
        double a[4][4];

        #pragma omp for schedule(dynamic, 1)
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
        {
            assert(lo_descriptor(idxlo).rsd_set.size() <= 4);

            int idxrf = atom_type_->indexr().index_by_idxlo(idxlo);

            if (lo_descriptor(idxlo).type == lo_rs)
            {
                // number of radial solutions
                int num_rs = (int)lo_descriptor(idxlo).rsd_set.size();

                std::vector< std::vector<double> > p(num_rs);
                std::vector< std::vector<double> > hp(num_rs);

                for (int order = 0; order < num_rs; order++)
                {
                    radial_solution_descriptor& rsd = lo_descriptor(idxlo).rsd_set[order];
                    
                    // find linearization energies
                    if (rsd.auto_enu == 1 || rsd.auto_enu == 2) 
                    {
                        solver.bound_state(rsd.n, rsd.l, spherical_potential_, rsd.enu, p[order]);
                        if (rsd.auto_enu == 2) rsd.enu = solver.find_enu(rsd.n, rsd.l, spherical_potential_, rsd.enu);
                    }

                    double dpdr;
                    solver.solve_in_mt(rsd.l, rsd.enu, rsd.dme, spherical_potential_, p[order], hp[order], dpdr); 
                    double p1 = p[order][nmtp - 1]; // save last value
                    
                    // normalize radial solutions and divide by r
                    for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(p[order][ir], 2);
                    double norm = 1.0 / sqrt(s.interpolate().integrate(0));

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        p[order][ir] *= (norm / atom_type_->radial_grid(ir));
                        hp[order][ir] *= norm; // don't divide hp by r
                        s[ir] = p[order][ir];
                    }
                    dpdr *= norm;
                    p1 *= norm;

                    s.interpolate();
                    
                    // compute radial derivatives
                    for (int dm = 0; dm < num_rs; dm++) a[order][dm] = s.deriv(dm, nmtp - 1);
                    a[order][1] = (dpdr - p1 / R) / R; // replace 1st derivative with more precise value

                }

                double b[] = {0.0, 0.0, 0.0, 0.0};
                b[num_rs - 1] = 1.0;

                int info = linalg<lapack>::gesv(num_rs, 1, &a[0][0], 4, b, 4);

                if (info) 
                {
                    std::stringstream s;
                    s << "gesv returned " << info;
                    error_local(__FILE__, __LINE__, s);
                }
                
                // take linear combination of radial solutions
                for (int order = 0; order < num_rs; order++)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        radial_functions_(ir, idxrf, 0) += b[order] * p[order][ir];
                        radial_functions_(ir, idxrf, 1) += b[order] * hp[order][ir];
                    }
                }

                // normalize
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(radial_functions_(ir, idxrf, 0), 2);
                double norm = 1.0 / sqrt(s.interpolate().integrate(2));

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf, 0) *= norm;
                    radial_functions_(ir, idxrf, 1) *= norm;
                }
                
                if (fabs(radial_functions_(nmtp - 1, idxrf, 0)) > 1e-10)
                {
                    std::stringstream s;
                    s << "atom symmetry class id : " << id() << " (" << atom_type()->symbol() << ")" << std::endl
                      << "local orbital " << idxlo << " is not zero at MT boundary" << std::endl 
                      << "  value : " << radial_functions_(nmtp - 1, idxrf, 0);
                    warning_local(__FILE__, __LINE__, s);
                }
            }

            if (lo_descriptor(idxlo).type == lo_cp)
            {
                int l = lo_descriptor(idxlo).l;
                int p1 = lo_descriptor(idxlo).p1;
                int p2 = lo_descriptor(idxlo).p2;
                double R = atom_type_->mt_radius();

                for (int ir = 0; ir < nmtp; ir++)
                {
                    double r = atom_type_->radial_grid(ir);
                    radial_functions_(ir, idxrf, 0) = Utils::confined_polynomial(r, R, p1, p2, 0);
                    radial_functions_(ir, idxrf, 1) = -0.5 * Utils::confined_polynomial(r, R, p1 + 1, p2, 2) +
                        (0.5 * l * (l + 1) / pow(r, 2) + spherical_potential_[ir]) * (radial_functions_(ir, idxrf, 0) * r);
                }
                
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(radial_functions_(ir, idxrf, 0), 2);
                double norm = 1.0 / sqrt(s.interpolate().integrate(2));

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf, 0) *= norm;
                    radial_functions_(ir, idxrf, 1) *= norm;
                }
            }
        }
    }
    
    if (debug_level >= 1 && num_lo_descriptors() > 0) check_lo_linear_independence();

    //if (verbosity_level > 0) dump_lo();
}

void Atom_symmetry_class::check_lo_linear_independence()
{
    int nmtp = atom_type_->num_mt_points();
    
    Spline<double> s(nmtp, atom_type_->radial_grid());
    mdarray<double, 2> loprod(num_lo_descriptors(), num_lo_descriptors());
    mdarray<complex16, 2> loprod_tmp(num_lo_descriptors(), num_lo_descriptors());
    for (int idxlo1 = 0; idxlo1 < num_lo_descriptors(); idxlo1++)
    {
        int idxrf1 = atom_type_->indexr().index_by_idxlo(idxlo1);
        
        for (int idxlo2 = 0; idxlo2 < num_lo_descriptors(); idxlo2++)
        {
            int idxrf2 = atom_type_->indexr().index_by_idxlo(idxlo2);
            
            for (int ir = 0; ir < nmtp; ir++)
                s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
            s.interpolate();
            
            if (lo_descriptor(idxlo1).l == lo_descriptor(idxlo2).l)
            {
                loprod(idxlo1, idxlo2) = s.integrate(2);
            }
            else
            {
                loprod(idxlo1, idxlo2) = 0.0;
            }
            loprod_tmp(idxlo1, idxlo2) = complex16(loprod(idxlo1, idxlo2), 0);
        }
    }
        
    standard_evp_lapack stdevp;

    std::vector<double> loprod_eval(num_lo_descriptors());
    mdarray<complex16, 2> loprod_evec(num_lo_descriptors(), num_lo_descriptors());

    stdevp.solve(num_lo_descriptors(), loprod_tmp.get_ptr(), loprod_tmp.ld(), &loprod_eval[0], 
                 loprod_evec.get_ptr(), loprod_evec.ld());

    if (fabs(loprod_eval[0]) < 0.001) 
    {
        printf("\n");
        printf("local orbitals for atom symmetry class %i are almost linearly dependent\n", id_);
        printf("local orbitals overlap matrix:\n");
        for (int i = 0; i < num_lo_descriptors(); i++)
        {
            for (int j = 0; j < num_lo_descriptors(); j++) printf("%12.6f", loprod(i, j));
            printf("\n");
        }
        printf("overlap matrix eigen-values:\n");
        for (int i = 0; i < num_lo_descriptors(); i++) printf("%12.6f", loprod_eval[i]);
        printf("\n");
    }
}

void Atom_symmetry_class::dump_lo()
{
    std::stringstream s;
    s << "local_orbitals_" << id_ << ".dat";
    FILE* fout = fopen(s.str().c_str(), "w");

    for (int ir = 0; ir <atom_type_->num_mt_points(); ir++)
    {
        fprintf(fout, "%f ", atom_type_->radial_grid(ir));
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
        {
            int idxrf = atom_type_->indexr().index_by_idxlo(idxlo);
            fprintf(fout, "%f ", radial_functions_(ir, idxrf, 0));
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    
    s.str("");
    s << "local_orbitals_h_" << id_ << ".dat";
    fout = fopen(s.str().c_str(), "w");

    for (int ir = 0; ir <atom_type_->num_mt_points(); ir++)
    {
        fprintf(fout, "%f ", atom_type_->radial_grid(ir));
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
        {
            int idxrf = atom_type_->indexr().index_by_idxlo(idxlo);
            fprintf(fout, "%f ", radial_functions_(ir, idxrf, 1));
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}

void Atom_symmetry_class::transform_radial_functions(bool ort_lo, bool ort_aw)
{
    Timer t("sirius::Atom_symmetry_class::transform_radial_functions");

    int nmtp = atom_type_->num_mt_points();
    Spline<double> s(nmtp, atom_type_->radial_grid());
    for (int l = 0; l <= atom_type_->indexr().lmax_lo(); l++)
    {
        // if we have local orbitals for the given l
        if ((atom_type_->indexr().num_lo(l) > 1) && ort_lo)
        {
            int naw = (atom_type_->num_aw_descriptors() > l) ? (int)atom_type_->aw_descriptor(l).size() : 0;

            // orthogonalize local orbitals
            for (int order1 = 1; order1 < atom_type_->indexr().num_lo(l); order1++)
            {
                int idxrf1 = atom_type_->indexr().index_by_l_order(l, naw + order1);

                for (int order2 = 0; order2 < order1; order2++)
                {
                    int idxrf2 = atom_type_->indexr().index_by_l_order(l, naw + order2);

                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                    double t1 = s.interpolate().integrate(2);
                        
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        radial_functions_(ir, idxrf1, 0) -= radial_functions_(ir, idxrf2, 0) * t1;
                        radial_functions_(ir, idxrf1, 1) -= radial_functions_(ir, idxrf2, 1) * t1;
                    }
                }
                    
                // normalize again
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(radial_functions_(ir, idxrf1, 0), 2);
                double norm = s.interpolate().integrate(2);

                if (fabs(norm) < 1e-10) 
                {
                    std::stringstream s;
                    s << "local orbital radial function for l = " << l << ", order = " << order1 << 
                         " is linearly dependent";
                    error_local(__FILE__, __LINE__, s);
                }

                norm = 1.0 / sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf1, 0) *= norm;
                    radial_functions_(ir, idxrf1, 1) *= norm;
                }
            }
        }
        
        if ((atom_type_->indexr().num_lo(l) > 0) && ort_aw)
        {
            int naw = (int)atom_type_->aw_descriptor(l).size();

            // orthogonalize aw functions to local orbitals
            for (int order1 = 0; order1 < naw; order1++)
            {
                int idxrf1 = atom_type_->indexr().index_by_l_order(l, order1);
                for (int order2 = 0; order2 < atom_type_->indexr().num_lo(l); order2++)
                {
                    int idxrf2 = atom_type_->indexr().index_by_l_order(l, naw + order2);
                    
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                    s.interpolate();
                    double t1 = s.integrate(1);
                        
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        radial_functions_(ir, idxrf1, 0) -= atom_type_->radial_grid(ir) * 
                                                            radial_functions_(ir, idxrf2, 0) * t1;
                        radial_functions_(ir, idxrf1, 1) -= radial_functions_(ir, idxrf2, 1) * t1;
                    }
                }
                    
                // normalize again
                for (int ir = 0; ir < nmtp; ir++) s[ir] = pow(radial_functions_(ir, idxrf1, 0), 2);
                double norm = s.interpolate().integrate(0);

                if (fabs(norm) < 1e-10) 
                    error_local(__FILE__, __LINE__, "aw radial function is linearly dependent");

                norm = 1.0 / sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++)
                {
                    radial_functions_(ir, idxrf1, 0) *= norm;
                    radial_functions_(ir, idxrf1, 1) *= norm;
                }

                error_local(__FILE__, __LINE__, "first, fix the radial derivative");
                
                // this is not precise
                double rderiv = (radial_functions_(nmtp - 1, idxrf1, 0) - radial_functions_(nmtp - 2, idxrf1, 0)) / 
                                atom_type_->radial_grid().dr(nmtp - 2);
                double R = atom_type_->mt_radius();

                aw_surface_derivatives_(order1, l) = (rderiv - radial_functions_(nmtp - 1, idxrf1, 0) / R) / R;
            }

        }
    }
   
    // divide by r
    for (int l = 0; l < num_aw_descriptors(); l++)
    {
        for (int order = 0; order < (int)aw_descriptor(l).size(); order++)
        {
            int idxrf = atom_type_->indexr().index_by_l_order(l, order);
            for (int ir = 0; ir < nmtp; ir++)
            {
                radial_functions_(ir, idxrf, 0) /= atom_type_->radial_grid(ir);
            }
        }
    }
}

void Atom_symmetry_class::initialize()
{
    aw_surface_derivatives_.set_dimensions(atom_type_->max_aw_order(), atom_type_->num_aw_descriptors());
    aw_surface_derivatives_.allocate();

    radial_functions_.set_dimensions(atom_type_->num_mt_points(), atom_type_->mt_radial_basis_size(), 2);
    radial_functions_.allocate();

    h_spherical_integrals_.set_dimensions(atom_type_->mt_radial_basis_size(), 
                                          atom_type_->mt_radial_basis_size());
    h_spherical_integrals_.allocate();
    
    o_radial_integrals_.set_dimensions(atom_type_->indexr().lmax() + 1, atom_type_->indexr().max_num_rf(), 
                                       atom_type_->indexr().max_num_rf());
    o_radial_integrals_.allocate();
    
    so_radial_integrals_.set_dimensions(atom_type_->indexr().lmax() + 1, atom_type_->indexr().max_num_rf(), 
                                        atom_type_->indexr().max_num_rf());
    so_radial_integrals_.allocate();

    // copy descriptors because enu is defferent between atom classes
    aw_descriptors_.resize(atom_type_->num_aw_descriptors());
    for (int i = 0; i < num_aw_descriptors(); i++) aw_descriptors_[i] = atom_type_->aw_descriptor(i);

    lo_descriptors_.resize(atom_type_->num_lo_descriptors());
    for (int i = 0; i < num_lo_descriptors(); i++) lo_descriptors_[i] = atom_type_->lo_descriptor(i);
    
    core_charge_density_.resize(atom_type_->radial_grid().size());
    memset(&core_charge_density_[0], 0, atom_type_->radial_grid().size() * sizeof(double));
}

void Atom_symmetry_class::set_spherical_potential(std::vector<double>& veff)
{
    int nmtp = atom_type_->num_mt_points();
    assert((int)veff.size() == nmtp);

    spherical_potential_.resize(atom_type_->radial_grid().size());
    
    // take current effective potential inside MT
    for (int ir = 0; ir < nmtp; ir++) spherical_potential_[ir] = veff[ir];

    // take potential of the free atom outside MT
    for (int ir = nmtp; ir < atom_type_->radial_grid().size(); ir++)
    {
        spherical_potential_[ir] = atom_type_->free_atom_potential(ir) - 
                                   (atom_type_->free_atom_potential(nmtp - 1) - veff[nmtp - 1]);
    }
}

void Atom_symmetry_class::generate_radial_functions()
{
    Timer t("sirius::Atom_symmetry_class::generate_radial_functions");

    radial_functions_.zero();

    generate_aw_radial_functions();
    generate_lo_radial_functions();
    transform_radial_functions(false, false);
    
    //** if (verbosity_level > 0)
    //** {
    //**     std::stringstream s;
    //**     s << "radial_functions_" << id_ << ".dat";
    //**     FILE* fout = fopen(s.str().c_str(), "w");

    //**     for (int ir = 0; ir <atom_type_->num_mt_points(); ir++)
    //**     {
    //**         fprintf(fout, "%f ", atom_type_->radial_grid(ir));
    //**         for (int idxrf = 0; idxrf < atom_type_->indexr().size(); idxrf++)
    //**         {
    //**             fprintf(fout, "%f ", radial_functions_(ir, idxrf, 0));
    //**         }
    //**         fprintf(fout, "\n");
    //**     }
    //**     fclose(fout);
    //** }
}

inline void Atom_symmetry_class::sync_radial_functions(int rank)
{
    // don't broadcast Hamiltonian radial functions, because they are used locally
    int size = (int)radial_functions_.size(0) * radial_functions_.size(1);
    Platform::bcast(radial_functions_.get_ptr(), size, rank);
    Platform::bcast(aw_surface_derivatives_.get_ptr(), (int)aw_surface_derivatives_.size(), rank);
}

inline void Atom_symmetry_class::sync_radial_integrals(int rank)
{
    Platform::bcast(h_spherical_integrals_.get_ptr(), (int)h_spherical_integrals_.size(), rank);
    Platform::bcast(o_radial_integrals_.get_ptr(), (int)o_radial_integrals_.size(), rank);
    Platform::bcast(so_radial_integrals_.get_ptr(), (int)so_radial_integrals_.size(), rank);
}

/** \todo OMP for radial integrals */
void Atom_symmetry_class::generate_radial_integrals()
{
    Timer t("sirius::Atom_symmetry_class::generate_radial_integrals");

    int nmtp = atom_type_->num_mt_points();

    h_spherical_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(nmtp, atom_type_->radial_grid()); 
        #pragma omp for
        for (int i1 = 0; i1 < atom_type_->mt_radial_basis_size(); i1++)
        {
            for (int i2 = 0; i2 < atom_type_->mt_radial_basis_size(); i2++)
            {
                // for spherical part of potential integrals are diagonal in l
                if (atom_type_->indexr(i1).l == atom_type_->indexr(i2).l)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        s[ir] = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 1) * 
                                atom_type_->radial_grid(ir); // hp was not divided by r, so we use r^{1}dr
                    }
                    h_spherical_integrals_(i1, i2) = s.interpolate().integrate(0) / y00;
                }
            }
        }
    }

    // TODO: kinetic energy for s-local orbitals? The best way to do it.
    // Problem: <s1 | \Delta | s2> != <s2 | \Delta | s1>  [???]

    // check and symmetrize local orbital radial integrals
    for (int idxlo1 = 0; idxlo1 < atom_type_->num_lo_descriptors(); idxlo1++)
    {
        int idxrf1 = atom_type_->indexr().index_by_idxlo(idxlo1);

        for (int idxlo2 = 0; idxlo2 <= idxlo1; idxlo2++)
        {
            int idxrf2 = atom_type_->indexr().index_by_idxlo(idxlo2);

            double diff = h_spherical_integrals_(idxrf1, idxrf2) - h_spherical_integrals_(idxrf2, idxrf1);

            if (debug_level >= 1 && fabs(diff) > 1e-8)
            {
                int l = atom_type_->indexr(idxrf2).l;
                std::stringstream s;
                s << "Wrong radial integrals between local orbitals " << idxlo1 << " and " << idxlo2 << " (l = " << l << ") " 
                  << " for atom class " << id() << std::endl
                  << "h(" << idxlo1 << "," << idxlo2 << ") = " << h_spherical_integrals_(idxrf1, idxrf2) << ","
                  << " h(" << idxlo2 << "," << idxlo1 << ") = " << h_spherical_integrals_(idxrf2, idxrf1) << ","
                  << " diff = " << diff;
                warning_local(__FILE__, __LINE__, s);
            }

            if (true)
            {
                double avg = 0.5 * (h_spherical_integrals_(idxrf1, idxrf2) + h_spherical_integrals_(idxrf2, idxrf1));
                h_spherical_integrals_(idxrf1, idxrf2) = avg;
                h_spherical_integrals_(idxrf2, idxrf1) = avg;
            }
        }
    }
        
    // check and symmetrize aw radial integrals
    for (int i2 = 0; i2 < atom_type_->mt_radial_basis_size() - atom_type_->num_lo_descriptors(); i2++)
    {
        int l = atom_type_->indexr(i2).l;
        int order2 = atom_type_->indexr(i2).order;
        
        for (int i1 = 0; i1 <= i2; i1++)
        {
            if (atom_type_->indexr(i1).l == l)
            {
                int order1 = atom_type_->indexr(i1).order;

                double R2 = pow(atom_type_->mt_radius(), 2);

                double surf12 = 0.5 * aw_surface_dm(l, order1, 0) * aw_surface_dm(l, order2, 1) * R2;
                double surf21 = 0.5 * aw_surface_dm(l, order2, 0) * aw_surface_dm(l, order1, 1) * R2;

                double v1 = y00 * h_spherical_integrals_(i1, i2) + surf12;
                double v2 = y00 * h_spherical_integrals_(i2, i1) + surf21; 

                double diff = fabs(v1 - v2);
                if (debug_level >= 1 && diff > 1e-8)
                {
                    std::stringstream s;
                    s << "Wrong augmented wave radial integrals for atom class " << id() << ", l = " << l << std::endl
                      << " order1 = " << order1 << ", value = " << v1 << std::endl
                      << " order2 = " << order2 << ", value = " << v2 << std::endl
                      << " <u_{l1,o1}| T |u_{l2,o2}> - <u_{l2,o2}| T |u_{l1,o1}> = " << diff << std::endl
                      << " h(1,2) = " << y00 * h_spherical_integrals_(i1, i2) << std::endl
                      << " h(2,1) = " << y00 * h_spherical_integrals_(i2, i1) << std::endl 
                      << " h(1,1) = " << y00 * h_spherical_integrals_(i1, i1) << std::endl
                      << " h(2,2) = " << y00 * h_spherical_integrals_(i2, i2) << std::endl 
                      << " surf_{12} = " << surf12 << std::endl
                      << " surf_{21} = " << surf21;
                    
                    warning_local(__FILE__, __LINE__, s);
                }

                if (true)
                {
                    double d = (v1 - v2);
                    h_spherical_integrals_(i1, i2) -= 0.5 * d / y00;
                    h_spherical_integrals_(i2, i1) += 0.5 * d / y00;

                }
            }
        }
    }
        
    o_radial_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(nmtp, atom_type_->radial_grid()); 
        #pragma omp for
        for (int l = 0; l <= atom_type_->indexr().lmax(); l++)
        {
            int nrf = atom_type_->indexr().num_rf(l);

            for (int order1 = 0; order1 < nrf; order1++)
            {
                int idxrf1 = atom_type_->indexr().index_by_l_order(l, order1);
                for (int order2 = 0; order2 < nrf; order2++)
                {
                    int idxrf2 = atom_type_->indexr().index_by_l_order(l, order2);

                    if (order1 == order2) 
                    {
                        o_radial_integrals_(l, order1, order2) = 1.0;
                    }
                    else
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                        o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(2);
                    }
                }
            }
        }
    }

    if (false) // TODO: if it's slow, compute only when spin-orbit is turned on
    {
        double soc = pow(2 * speed_of_light, -2);

        Spline<double> s(nmtp, atom_type_->radial_grid()); 
        Spline<double> s1(nmtp, atom_type_->radial_grid()); 
        Spline<double> ve(nmtp, atom_type_->radial_grid()); 
        
        for (int i = 0; i < nmtp; i++) ve[i] = spherical_potential_[i] + atom_type_->zn() / atom_type_->radial_grid(i);
        ve.interpolate();

        so_radial_integrals_.zero();
        for (int l = 0; l <= atom_type_->indexr().lmax(); l++)
        {
            int nrf = atom_type_->indexr().num_rf(l);

            for (int order1 = 0; order1 < nrf; order1++)
            {
                int idxrf1 = atom_type_->indexr().index_by_l_order(l, order1);
                for (int order2 = 0; order2 < nrf; order2++)
                {
                    int idxrf2 = atom_type_->indexr().index_by_l_order(l, order2);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double M = 1.0 - 2 * soc * spherical_potential_[ir];
                        // first part <f| dVe / dr |f'>
                        s[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * 
                                soc * ve.deriv(1, ir) / pow(M, 2);

                        // second part <f| d(z/r) / dr |f'>
                        s1[ir] = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) *
                                 soc * atom_type_->zn() / pow(M, 2);
                    }
                    s.interpolate();
                    s1.interpolate();

                    so_radial_integrals_(l, order1, order2) = s.integrate(1) + s1.integrate(-1);
                }
            }
        }
    }
}

void Atom_symmetry_class::write_enu(pstdout& pout)
{
    pout.printf("Atom : %s, class id : %i\n", atom_type_->symbol().c_str(), id_); 
    pout.printf("augmented waves\n");
    for (int l = 0; l < num_aw_descriptors(); l++)
    {
        for (int order = 0; order < (int)aw_descriptor(l).size(); order++)
        {
            radial_solution_descriptor& rsd = aw_descriptor(l)[order];
            pout.printf("n = %2i   l = %2i   order = %i   enu = %12.6f\n", rsd.n, rsd.l, order, rsd.enu);
        }
    }

    pout.printf("local orbitals\n");
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++)
    {
        if (lo_descriptor(idxlo).type == lo_rs)
        {
            for (int order = 0; order < (int)lo_descriptor(idxlo).rsd_set.size(); order++)
            {
                radial_solution_descriptor& rsd = lo_descriptor(idxlo).rsd_set[order];
                pout.printf("n = %2i   l = %2i   order = %i   enu = %12.6f\n", rsd.n, rsd.l, order, rsd.enu);
            }
        }
    }
    pout.printf("\n");
}

double Atom_symmetry_class::aw_surface_dm(int l, int order, int dm)
{
    switch (dm)
    {
        case 0:
        {
            int idxrf = atom_type_->indexr().index_by_l_order(l, order);
            return radial_functions_(atom_type_->num_mt_points() - 1, idxrf, 0);
        }
        case 1:
        {
            return aw_surface_derivatives_(order, l);
        }
        default:
        {
            error_local(__FILE__, __LINE__, "wrong order of radial derivative");
        }
    }

    return 0.0; // just to make compiler happy
}

void Atom_symmetry_class::generate_core_charge_density()
{
    Timer t("sirius::Atom_symmetry_class::generate_core_charge_density");

    if (atom_type_->num_core_electrons() == 0.0) return;
    
    //Radial_solver solver(true, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
    Radial_solver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
    
    Spline<double> rho(atom_type_->radial_grid().size(), atom_type_->radial_grid());
    
    std::vector<double> level_energy(atom_type_->num_atomic_levels());

    for (int ist = 0; ist < atom_type_->num_atomic_levels(); ist++)
        level_energy[ist] = -1.0 * atom_type_->zn() / 2 / pow(double(atom_type_->atomic_level(ist).n), 2);
    
    memset(&rho[0], 0, rho.num_points() * sizeof(double));
    #pragma omp parallel default(shared)
    {
        std::vector<double> rho_t(rho.num_points());
        memset(&rho_t[0], 0, rho.num_points() * sizeof(double));
        std::vector<double> p;
        
        #pragma omp for
        for (int ist = 0; ist < atom_type_->num_atomic_levels(); ist++)
        {
            if (atom_type_->atomic_level(ist).core)
            {
                solver.bound_state(atom_type_->atomic_level(ist).n, atom_type_->atomic_level(ist).l, 
                                   spherical_potential_, level_energy[ist], p);
        
                for (int i = 0; i < atom_type_->radial_grid().size(); i++)
                {
                    rho_t[i] += atom_type_->atomic_level(ist).occupancy * 
                                pow(y00 * p[i] / atom_type_->radial_grid(i), 2);
                }
            }
        }

        #pragma omp critical
        for (int i = 0; i < rho.num_points(); i++) rho[i] += rho_t[i];
    } 
        
    core_charge_density_ = rho.data_points();
    rho.interpolate();

    Spline<double> rho_mt(atom_type_->num_mt_points(), atom_type_->radial_grid());
    rho_mt.interpolate(core_charge_density_);

    core_leakage_ = fourpi * (rho.integrate(2) - rho_mt.integrate(2));

    core_eval_sum_ = 0.0;
    for (int ist = 0; ist < atom_type_->num_atomic_levels(); ist++)
    {
        if (atom_type_->atomic_level(ist).core)
        {
            assert(level_energy[ist] == level_energy[ist]);
            core_eval_sum_ += level_energy[ist] * atom_type_->atomic_level(ist).occupancy;
        }
    }
    assert(core_eval_sum_ == core_eval_sum_);
}

inline void Atom_symmetry_class::sync_core_charge_density(int rank)
{
    assert(core_charge_density_.size() != 0);
    assert(&core_charge_density_[0] != NULL);

    Platform::bcast(&core_charge_density_[0], atom_type_->radial_grid().size(), rank);
    Platform::bcast(&core_leakage_, 1, rank);
    Platform::bcast(&core_eval_sum_, 1, rank);
}

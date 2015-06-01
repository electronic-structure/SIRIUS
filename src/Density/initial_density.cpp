#include "density.h"

namespace sirius {

void Density::initial_density()
{
    Timer t("sirius::Density::initial_density");

    zero();
    
    auto rl = ctx_.reciprocal_lattice();

    /* let rho(r) = a*Exp[-b*r]
       \int rho(r) dr = 4pi * a * \int Exp[-b*r]*r*r*dr = 4pi*a * (2/b^3) = Z
       a = Z * b^3 / 2 / 4 / pi
    */

    if (unit_cell_.full_potential())
    {
        /* initialize smooth density of free atoms */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) unit_cell_.atom_type(iat)->init_free_atom(true);

        //== /* compute radial integrals */
        auto rho_radial_integrals = generate_rho_radial_integrals(0);
        //auto rho_radial_integrals = generate_rho_radial_integrals(5);

        /* compute contribution from free atoms to the interstitial density */
        auto v = rl->make_periodic_function(rho_radial_integrals, rl->num_gvec());
        
        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z = mdarray<double_complex, 1>(&v[0], rl->num_gvec()).checksum();
        DUMP("checksum(rho_pw): %18.10f %18.10f", std::real(z), std::imag(z));
        #endif

        /* set plane-wave coefficients of the charge density */
        memcpy(&rho_->f_pw(0), &v[0], rl->num_gvec() * sizeof(double_complex));
        /* convert charge deisnty to real space mesh */
        rho_->fft_transform(1);

        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z2 = rho_->f_it().checksum(); 
        DUMP("checksum(rho_it): %18.10f %18.10f", std::real(z2), std::imag(z2));
        #endif

        #ifdef _PRINT_OBJECT_HASH_
        DUMP("hash(rhopw): %16llX", rho_->f_pw().hash());
        DUMP("hash(rhoit): %16llX", rho_->f_it().hash());
        #endif

        /* remove possible negative noise */
        for (int ir = 0; ir < fft_->size(); ir++)
        {
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }

        int ngv_loc = (int)rl->spl_num_gvec().local_size();

        /* mapping between G-shell (global index) and a list of G-vectors (local index) */
        std::map<int, std::vector<int> > gsh_map;

        for (int igloc = 0; igloc < ngv_loc; igloc++)
        {
            /* global index of the G-vector */
            int ig = (int)rl->spl_num_gvec(igloc);
            /* index of the G-vector shell */
            int igsh = rl->gvec_shell(ig);
            if (gsh_map.count(igsh) == 0) gsh_map[igsh] = std::vector<int>();
            gsh_map[igsh].push_back(igloc);
        }

        /* list of G-shells for the curent MPI rank */
        std::vector<std::pair<int, std::vector<int> > > gsh_list;
        for (auto& i: gsh_map) gsh_list.push_back(std::pair<int, std::vector<int> >(i.first, i.second));

        int lmax = parameters_.lmax_rho();
        int lmmax = Utils::lmmax(lmax);
        
        sbessel_approx sba(&unit_cell_, lmax, rl->gvec_shell_len(1), rl->gvec_shell_len(rl->num_gvec_shells_inner() - 1), 1e-6);
        
        std::vector<double> gvec_len(gsh_list.size());
        for (int i = 0; i < (int)gsh_list.size(); i++)
        {
            gvec_len[i] = rl->gvec_shell_len(gsh_list[i].first);
        }
        sba.approximate(gvec_len);

        auto l_by_lm = Utils::l_by_lm(lmax);

        std::vector<double_complex> zil(lmax + 1);
        for (int l = 0; l <= lmax; l++) zil[l] = pow(double_complex(0, 1), l);

        Timer t3("sirius::Density::initial_density|znulm");

        mdarray<double_complex, 3> znulm(sba.nqnu_max(), lmmax, unit_cell_.num_atoms());
        znulm.zero();
        
        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int iat = unit_cell_.atom(ia)->type_id();

            /* loop over local fraction of G-shells */
            for (int i = 0; i < (int)gsh_list.size(); i++)
            {
                auto& gv = gsh_list[i].second;
                
                /* loop over G-vectors */
                for (int igloc: gv)
                {
                    /* global index of the G-vector */
                    int ig = rl->spl_num_gvec(igloc);

                    double_complex z1 = rl->gvec_phase_factor<local>(igloc, ia) * v[ig] * fourpi; 

                    for (int lm = 0; lm < lmmax; lm++)
                    {
                        int l = l_by_lm[lm];
                        
                        /* number of expansion coefficients */
                        int nqnu = sba.nqnu(l, iat);

                        double_complex z2 = z1 * zil[l] * rl->gvec_ylm(lm, igloc);
                    
                        for (int iq = 0; iq < nqnu; iq++) znulm(iq, lm, ia) += z2 * sba.coeff(iq, i, l, iat);
                    }
                }
            }
        }
        ctx_.comm().allreduce(znulm.at<CPU>(), (int)znulm.size());
        t3.stop();
        
        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z3 = znulm.checksum();
        DUMP("checksum(znulm): %18.10f %18.10f", std::real(z3), std::imag(z3));
        #endif

        Timer t4("sirius::Density::initial_density|rholm");
        
        SHT sht(lmax);

        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int iat = unit_cell_.atom(ia)->type_id();
            //double b = 4;
            //double R = parameters_.unit_cell()->atom(ia)->mt_radius();
            //int Z = parameters_.unit_cell()->atom(ia)->zn();

            //double c2 = (std::pow(b,3)*(3 + b*R)*Z)/(8.*std::exp(b*R)*pi*std::pow(R,2));
            //double c3 = -(std::pow(b,3)*(2 + b*R)*Z)/(8.*std::exp(b*R)*pi*std::pow(R,3));

            Spheric_function<spectral, double_complex> rhoylm(lmmax, unit_cell_.atom(ia)->radial_grid());
            rhoylm.zero();
            #pragma omp parallel for
            for (int lm = 0; lm < lmmax; lm++)
            {
                int l = l_by_lm[lm];
                for (int iq = 0; iq < sba.nqnu(l, iat); iq++)
                {
                    double qnu = sba.qnu(iq, l, iat);

                    for (int ir = 0; ir < unit_cell_.atom(ia)->num_mt_points(); ir++)
                    {
                        double x = unit_cell_.atom(ia)->radial_grid(ir);
                        rhoylm(lm, ir) += znulm(iq, lm, ia) * gsl_sf_bessel_jl(l, x * qnu);
                    }
                }
            }
            for (int ir = 0; ir < unit_cell_.atom(ia)->num_mt_points(); ir++)
            {
                double x = unit_cell_.atom(ia)->radial_grid(ir);
                rhoylm(0, ir) += (v[0] - unit_cell_.atom(ia)->type()->free_atom_density(x)) / y00;
                //rhoylm(0, ir) += (v[0] - (c2 * std::pow(x, 2) + c3*std::pow(x, 3))) / y00;
            }
            sht.convert(rhoylm, rho_->f_mt(ialoc));
        }
        
        t4.stop();

        /* initialize density of free atoms (not smoothed) */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) unit_cell_.atom_type(iat)->init_free_atom(false);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            //double b = 8;
            //int Z = parameters_.unit_cell()->atom(ia)->zn();

            auto p = unit_cell_.spl_num_atoms().location(ia);
            
            if (p.second == ctx_.comm().rank())
            {
                /* add density of a free atom */
                for (int ir = 0; ir < unit_cell_.atom(ia)->num_mt_points(); ir++)
                {
                    double x = unit_cell_.atom(ia)->type()->radial_grid(ir);
                    rho_->f_mt<local>(0, ir, (int)p.first) += unit_cell_.atom(ia)->type()->free_atom_density(x) / y00;
                    //rho_->f_mt<local>(0, ir, (int)p.first) += Z * std::pow(b, 3) * std::exp(-b * x) / 8 / pi / y00;
                }
            }
        }

        /* initialize the magnetization */
        if (parameters_.num_mag_dims())
        {
            for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
            {
                int ia = (int)unit_cell_.spl_num_atoms(ialoc);
                vector3d<double> v = unit_cell_.atom(ia)->vector_field();
                double len = v.length();

                int nmtp = unit_cell_.atom(ia)->type()->num_mt_points();
                Spline<double> rho(unit_cell_.atom(ia)->type()->radial_grid());
                double R = unit_cell_.atom(ia)->type()->mt_radius();
                for (int ir = 0; ir < nmtp; ir++)
                {
                    double x = unit_cell_.atom(ia)->type()->radial_grid(ir);
                    rho[ir] = rho_->f_mt<local>(0, ir, ialoc) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
                }

                /* maximum magnetization which can be achieved if we smooth density towards MT boundary */
                double q = fourpi * rho.interpolate().integrate(2);
                
                /* if very strong initial magnetization is given */
                if (q < len)
                {
                    /* renormalize starting magnetization */
                    for (int x = 0; x < 3; x++) v[x] *= (q / len);

                    len = q;
                }

                if (len > 1e-8)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[0]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[2] / q / y00;

                    if (parameters_.num_mag_dims() == 3)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                        {
                            magnetization_[1]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[0] / q / y00;
                            magnetization_[2]->f_mt<local>(0, ir, ialoc) = rho[ir] * v[1] / q / y00;
                        }
                    }
                }
            }
        }
    }

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential) 
    {
        auto rho_radial_integrals = generate_rho_radial_integrals(1);
        #ifdef _WRITE_OBJECTS_HASH_
        DUMP("hash(rho_radial_integrals) : %16llX", rho_radial_integrals.hash());
        #endif

        std::vector<double_complex> v = rl->make_periodic_function(rho_radial_integrals, rl->num_gvec());
        #ifdef _WRITE_OBJECTS_HASH_
        DUMP("hash(rho(G)) : %16llX", Utils::hash(&v[0], rl->num_gvec() * sizeof(double_complex)));
        #endif

        memcpy(&rho_->f_pw(0), &v[0], rl->num_gvec() * sizeof(double_complex));

        double charge = real(rho_->f_pw(0) * unit_cell_.omega());
        if (std::abs(charge - unit_cell_.num_valence_electrons()) > 1e-6)
        {
            std::stringstream s;
            s << "wrong initial charge density" << std::endl
              << "  integral of the density : " << real(rho_->f_pw(0) * unit_cell_.omega()) << std::endl
              << "  target number of electrons : " << unit_cell_.num_valence_electrons();
            warning_global(__FILE__, __LINE__, s);
        }

        fft_->input(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_->f_it<global>(0));

        #ifdef _WRITE_OBJECTS_HASH_
        DUMP("hash(rho(r)) : %16llX", Utils::hash(&rho_->f_it<global>(0), fft_->size() * sizeof(double)));
        #endif
        
        /* remove possible negative noise */
        for (int ir = 0; ir < fft_->size(); ir++)
        {
            rho_->f_it<global>(ir) = rho_->f_it<global>(ir) *  unit_cell_.num_valence_electrons() / charge;
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }

        #ifdef _WRITE_OBJECTS_HASH_
        DUMP("hash(rho) : %16llX", rho_->hash());
        #endif

        //== FILE* fout = fopen("unit_cell.xsf", "w");
        //== fprintf(fout, "CRYSTAL\n");
        //== fprintf(fout, "PRIMVEC\n");
        //== auto& lv = unit_cell_.lattice_vectors();
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "CONVVEC\n");
        //== for (int i = 0; i < 3; i++)
        //== {
        //==     fprintf(fout, "%18.12f %18.12f %18.12f\n", lv(0, i), lv(1, i), lv(2, i));
        //== }
        //== fprintf(fout, "PRIMCOORD\n");
        //== fprintf(fout, "%i 1\n", unit_cell_.num_atoms());
        //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //== {
        //==     auto pos = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia)->position());
        //==     fprintf(fout, "%i %18.12f %18.12f %18.12f\n", unit_cell_.atom(ia)->zn(), pos[0], pos[1], pos[2]);
        //== }
        //== fclose(fout);


        //== /* initialize the magnetization */
        //== if (parameters_.num_mag_dims())
        //== {
        //==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //==     {
        //==         vector3d<double> v = unit_cell_.atom(ia)->vector_field();
        //==         double len = v.length();

        //==         for (int j0 = 0; j0 < fft_->size(0); j0++)
        //==         {
        //==             for (int j1 = 0; j1 < fft_->size(1); j1++)
        //==             {
        //==                 for (int j2 = 0; j2 < fft_->size(2); j2++)
        //==                 {
        //==                     /* get real space fractional coordinate */
        //==                     vector3d<double> v0(double(j0) / fft_->size(0), double(j1) / fft_->size(1), double(j2) / fft_->size(2));
        //==                     /* index of real space point */
        //==                     int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));

        //==                     for (int t0 = -1; t0 <= 1; t0++)
        //==                     {
        //==                         for (int t1 = -1; t1 <= 1; t1++)
        //==                         {
        //==                             for (int t2 = -1; t2 <= 1; t2++)
        //==                             {
        //==                                 vector3d<double> v1 = v0 - (unit_cell_.atom(ia)->position() + vector3d<double>(t0, t1, t2));
        //==                                 auto r = unit_cell_.get_cartesian_coordinates(vector3d<double>(v1));
        //==                                 if (r.length() <= 2.0)
        //==                                 {
        //==                                     magnetization_[0]->f_it<global>(ir) = 1.0;
        //==                                 }
        //==                             }
        //==                         }
        //==                     }
        //==                 }
        //==             }
        //==         }
        //==     }
        //== }


        //== mdarray<double, 3> rho_grid(&rho_->f_it<global>(0), fft_->size(0), fft_->size(1), fft_->size(2));
        //== mdarray<double, 4> pos_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));

        //== mdarray<double, 4> mag_grid(3, fft_->size(0), fft_->size(1), fft_->size(2));
        //== mag_grid.zero();

        //== // loop over 3D array (real space)
        //== for (int j0 = 0; j0 < fft_->size(0); j0++)
        //== {
        //==     for (int j1 = 0; j1 < fft_->size(1); j1++)
        //==     {
        //==         for (int j2 = 0; j2 < fft_->size(2); j2++)
        //==         {
        //==             int ir = static_cast<int>(j0 + j1 * fft_->size(0) + j2 * fft_->size(0) * fft_->size(1));
        //==             // get real space fractional coordinate
        //==             double frv[] = {double(j0) / fft_->size(0), 
        //==                             double(j1) / fft_->size(1), 
        //==                             double(j2) / fft_->size(2)};
        //==             vector3d<double> rv = parameters_.unit_cell()->get_cartesian_coordinates(vector3d<double>(frv));
        //==             for (int x = 0; x < 3; x++) pos_grid(x, j0, j1, j2) = rv[x];
        //==             if (parameters_.num_mag_dims() == 1) mag_grid(2, j0, j1, j2) = magnetization_[0]->f_it<global>(ir);
        //==             if (parameters_.num_mag_dims() == 3) 
        //==             {
        //==                 mag_grid(0, j0, j1, j2) = magnetization_[1]->f_it<global>(ir);
        //==                 mag_grid(1, j0, j1, j2) = magnetization_[2]->f_it<global>(ir);
        //==             }
        //==         }
        //==     }
        //== }

        //== HDF5_tree h5_rho("rho.hdf5", true);
        //== h5_rho.write("rho", rho_grid);
        //== h5_rho.write("pos", pos_grid);
        //== h5_rho.write("mag", mag_grid);

        //== FILE* fout = fopen("rho.xdmf", "w");
        //== //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //== //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //== //==               "<Xdmf>\n"
        //== //==               "  <Domain Name=\"name1\">\n"
        //== //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //== //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //== //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //== //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //== //==               "      </Geometry>\n"
        //== //==               "      <Attribute\n"
        //== //==               "           AttributeType=\"Scalar\"\n"
        //== //==               "           Center=\"Node\"\n"
        //== //==               "           Name=\"rho\">\n"
        //== //==               "          <DataItem\n"
        //== //==               "             NumberType=\"Float\"\n"
        //== //==               "             Precision=\"8\"\n"
        //== //==               "             Dimensions=\"%i %i %i\"\n"
        //== //==               "             Format=\"HDF\">\n"
        //== //==               "             rho.hdf5:/rho\n"
        //== //==               "          </DataItem>\n"
        //== //==               "        </Attribute>\n"
        //== //==               "    </Grid>\n"
        //== //==               "  </Domain>\n"
        //== //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fprintf(fout, "<?xml version=\"1.0\" ?>\n"
        //==               "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\">\n"
        //==               "<Xdmf>\n"
        //==               "  <Domain Name=\"name1\">\n"
        //==               "    <Grid Name=\"fft_fine_grid\" Collection=\"Unknown\">\n"
        //==               "      <Topology TopologyType=\"3DSMesh\" NumberOfElements=\" %i %i %i \"/>\n"
        //==               "      <Geometry GeometryType=\"XYZ\">\n"
        //==               "        <DataItem Dimensions=\"%i %i %i 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">rho.hdf5:/pos</DataItem>\n"
        //==               "      </Geometry>\n"
        //==               "      <Attribute\n"
        //==               "           AttributeType=\"Vector\"\n"
        //==               "           Center=\"Node\"\n"
        //==               "           Name=\"mag\">\n"
        //==               "          <DataItem\n"
        //==               "             NumberType=\"Float\"\n"
        //==               "             Precision=\"8\"\n"
        //==               "             Dimensions=\"%i %i %i 3\"\n"
        //==               "             Format=\"HDF\">\n"
        //==               "             rho.hdf5:/mag\n"
        //==               "          </DataItem>\n"
        //==               "        </Attribute>\n"
        //==               "    </Grid>\n"
        //==               "  </Domain>\n"
        //==               "</Xdmf>\n", fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2), fft_->size(0), fft_->size(1), fft_->size(2));
        //== fclose(fout);
        
        fft_->input(&rho_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(fft_->num_gvec(), fft_->index_map(), &rho_->f_pw(0));
    }

    rho_->sync(true, true);
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->sync(true, true);

    if (unit_cell_.full_potential())
    {
        #ifdef _PRINT_OBJECT_CHECKSUM_
        DUMP("checksum(rhomt): %18.10f", rho_->f_mt().checksum());
        #endif

        #ifdef _PRINT_OBJECT_HASH_
        DUMP("hash(rho) : %16llX", rho_->hash());
        #endif
        /* check initial charge */
        std::vector<double> nel_mt;
        double nel_it;
        double nel = rho_->integrate(nel_mt, nel_it);
        if (unit_cell_.num_electrons() > 1e-8 && std::abs(nel - unit_cell_.num_electrons()) / unit_cell_.num_electrons()  > 1e-3)
        {
            std::stringstream s;
            s << "wrong initial charge density" << std::endl
              << "  integral of the density : " << nel << std::endl
              << "  target number of electrons : " << unit_cell_.num_electrons();
            warning_global(__FILE__, __LINE__, s);
        }
    }
}

};

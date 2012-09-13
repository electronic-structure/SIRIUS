#ifndef __ATOM_SYMMETRY_CLASS_H__
#define __ATOM_SYMMETRY_CLASS_H__

namespace sirius {

struct radial_solution
{
    radial_solution_descriptor rsd;

    std::vector<double> p;
    std::vector<double> hp;
};

typedef std::vector<radial_solution> radial_solution_set;

class AtomSymmetryClass
{
    private:
        
        int id_;

        std::vector<int> atom_id_;
        
        AtomType* atom_type_;

        std::vector<double> spherical_potential_;

        std::vector<radial_solution_set> aw_radial_solutions_;
        
        std::vector<radial_solution_set> lo_radial_solutions_;



        mdarray<double,3> radial_functions_;
        
        mdarray<double,3> lo_radial_functions_;

        mdarray<double,4> aw_radial_functions_;

        mdarray<double,2> aw_surface_derivatives_;
        
    public:
    
        AtomSymmetryClass(int id_, 
                          AtomType* atom_type_) : id_(id_),
                                                  atom_type_(atom_type_)
        {
        
        }

        void init()
        {
            aw_radial_solutions_.clear();
            lo_radial_solutions_.clear();

            assert(atom_type_->num_aw_descriptors() != 0);

            for (int i = 0; i < atom_type_->num_aw_descriptors(); i++)
            {
                radial_solution_descriptor_set rsds = atom_type_->aw_descriptor(i);
                
                radial_solution_set rss(rsds.size());
                for (int order = 0; order < (int)rsds.size(); order++)
                   rss[order].rsd = rsds[order];
                
                aw_radial_solutions_.push_back(rss);
            }
            
            for (int i = 0; i < atom_type_->num_lo_descriptors(); i++)
            {
                radial_solution_descriptor_set rsds = atom_type_->lo_descriptor(i);
                
                radial_solution_set rss(rsds.size());
                for (int order = 0; order < (int)rsds.size(); order++)
                   rss[order].rsd = rsds[order];
                
                lo_radial_solutions_.push_back(rss);
            }

            aw_radial_functions_.set_dimensions(atom_type_->num_mt_points(), atom_type_->max_aw_order(), aw_radial_solutions_.size(), 2);
            aw_radial_functions_.allocate();

            aw_surface_derivatives_.set_dimensions(atom_type_->max_aw_order(), aw_radial_solutions_.size());
            aw_surface_derivatives_.allocate();

            lo_radial_functions_.set_dimensions(atom_type_->num_mt_points(), lo_radial_solutions_.size(), 2);
            lo_radial_functions_.allocate();

            radial_functions_.set_dimensions(atom_type_->num_mt_points(), atom_type_->indexr().size(), 2);
            radial_functions_.allocate();
        }

        inline int id()
        {
            return id_;
        }

        inline void add_atom_id(int _atom_id)
        {
            atom_id_.push_back(_atom_id);
        }
        
        inline int num_atoms()
        {
            return atom_id_.size();
        }

        inline int atom_id(int idx)
        {
            return atom_id_[idx];
        }

        void set_spherical_potential(std::vector<double>& veff)
        {
            int nmtp = atom_type_->num_mt_points();
            assert((int)veff.size() == nmtp);

            spherical_potential_.resize(atom_type_->radial_grid().size());
            
            // take current effective potential inside MT
            for (int ir = 0; ir < nmtp; ir++)
                spherical_potential_[ir] = veff[ir];

            // take potential of the free atom outside MT
            for (int ir = nmtp; ir < atom_type_->radial_grid().size(); ir++)
                spherical_potential_[ir] = atom_type_->free_atom_potential(ir) - (atom_type_->free_atom_potential(nmtp - 1) - veff[nmtp - 1]);
#if 0
            int nmtp = atom_type_->num_mt_points();

            std::ofstream out("veff_for_enu.dat");

            for (int ir = 0; ir < nmtp; ir++)
               out << atom_type_->radial_grid()[ir] << " " << spherical_potential_[ir] << std::endl;
            out << std::endl;

            for (int ir = 0; ir < atom_type_->radial_grid().size(); ir++)
                out << atom_type_->radial_grid()[ir] << " " << atom_type_->free_atom_potential(ir) << std::endl;
            out << std::endl;
            
            for (int ir = 0; ir < atom_type_->radial_grid().size(); ir++)
                out << atom_type_->radial_grid()[ir] << " " << veff[ir] << std::endl;
            out << std::endl;

            out.close();
#endif
        }

        void generate_aw_radial_functions()
        {
            int nmtp = atom_type_->num_mt_points();
            Spline<double> s(nmtp, atom_type_->radial_grid());
            RadialSolver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
            
            for (int i = 0; i < (int)aw_radial_solutions_.size(); i++)
            {
                for (int order = 0; order < (int)aw_radial_solutions_[i].size(); order++)
                {
                    // find linearization energies
                    if (aw_radial_solutions_[i][order].rsd.auto_enu)
                    {
                        std::vector<double> p;
                        solver.bound_state(aw_radial_solutions_[i][order].rsd.n, aw_radial_solutions_[i][order].rsd.l, 
                                           spherical_potential_, aw_radial_solutions_[i][order].rsd.enu, p);
                    }

                    solver.solve_in_mt(aw_radial_solutions_[i][order].rsd.l, aw_radial_solutions_[i][order].rsd.enu, 
                                       aw_radial_solutions_[i][order].rsd.dme, spherical_potential_, 
                                       aw_radial_solutions_[i][order].p, aw_radial_solutions_[i][order].hp);
                    // normalize
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = aw_radial_solutions_[i][order].p[ir] * aw_radial_solutions_[i][order].p[ir];
                    s.interpolate();
                    double norm = s.integrate(0);
                    norm = 1.0 / sqrt(norm);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        aw_radial_functions_(ir, order, i, 0) = aw_radial_solutions_[i][order].p[ir] * norm;
                        aw_radial_functions_(ir, order, i, 1) = aw_radial_solutions_[i][order].hp[ir] * norm;
                    }
                    
                    // orthogonalize
                    for (int order1 = 0; order1 < order - 1; order1++)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = aw_radial_functions_(ir, order, i, 0) * aw_radial_functions_(ir, order1, i, 0);
                        s.interpolate();
                        double t1 = s.integrate(0);
                        
                        for (int ir = 0; ir < nmtp; ir++)
                        {
                            aw_radial_functions_(ir, order, i, 0) -= aw_radial_functions_(ir, order1, i, 0) * t1;
                            aw_radial_functions_(ir, order, i, 1) -= aw_radial_functions_(ir, order1, i, 1) * t1;
                        }
                    }
                        
                    // normalize again
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = aw_radial_functions_(ir, order, i, 0) * aw_radial_functions_(ir, order, i, 0);
                    s.interpolate();
                    norm = s.integrate(0);

                    if (fabs(norm) < 1e-12)
                        error(__FILE__, __LINE__, "aw radial functions are linearly dependent");

                    norm = 1.0 / sqrt(norm);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        aw_radial_functions_(ir, order, i, 0) *= norm;
                        aw_radial_functions_(ir, order, i, 1) *= norm;
                    }

                    // radial derivative
                    double rderiv = (aw_radial_functions_(nmtp - 1, order, i, 0) - 
                                     aw_radial_functions_(nmtp - 2, order, i, 0)) / atom_type_->radial_grid().dr(nmtp - 2);
                    double R = atom_type_->mt_radius();

                    aw_surface_derivatives_(order, i) = (rderiv - aw_radial_functions_(nmtp - 1, order, i, 0) / R) / R;
                    
                    // divide by r
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        aw_radial_functions_(ir, order, i, 0) /= atom_type_->radial_grid()[ir];
                        aw_radial_functions_(ir, order, i, 1) /= atom_type_->radial_grid()[ir];
                    }
                }
            }
        }

        void generate_lo_radial_functions()
        {
            int nmtp = atom_type_->num_mt_points();
            Spline<double> s(nmtp, atom_type_->radial_grid());
            RadialSolver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());
            
            lo_radial_functions_.zero();
            double a[4][4];

            for (int i = 0; i < (int)lo_radial_solutions_.size(); i++)
            {
                assert(lo_radial_solutions_[i].size() < 4);
                
                for (int order = 0; order < (int)lo_radial_solutions_[i].size(); order++)
                {
                    // find linearization energies
                    if (lo_radial_solutions_[i][order].rsd.auto_enu)
                    {
                        std::vector<double> p;
                        solver.bound_state(lo_radial_solutions_[i][order].rsd.n, lo_radial_solutions_[i][order].rsd.l, 
                                           spherical_potential_, lo_radial_solutions_[i][order].rsd.enu, p);
                    }

                    solver.solve_in_mt(lo_radial_solutions_[i][order].rsd.l, lo_radial_solutions_[i][order].rsd.enu, 
                                       lo_radial_solutions_[i][order].rsd.dme, spherical_potential_, 
                                       lo_radial_solutions_[i][order].p, lo_radial_solutions_[i][order].hp);
                    
                    // normalize radial solutions and divide by r
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = lo_radial_solutions_[i][order].p[ir] * lo_radial_solutions_[i][order].p[ir];
                    s.interpolate();
                    double norm = s.integrate(0);
                    norm = 1.0 / sqrt(norm);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        lo_radial_solutions_[i][order].p[ir] *= (norm / atom_type_->radial_grid()[ir]);
                        lo_radial_solutions_[i][order].hp[ir] *= (norm / atom_type_->radial_grid()[ir]);
                        s[ir] = lo_radial_solutions_[i][order].p[ir];
                    }

                    // compute radial derivatives
                    s.interpolate();

                    for (int dm = 0; dm < (int)lo_radial_solutions_[i].size(); dm++)
                        a[order][dm] = s.deriv(dm, nmtp - 1);
                }
                double b[] = {0.0, 0.0, 0.0, 0.0};
                b[lo_radial_solutions_[i].size() - 1] = 1.0;

                int info = gesv<double>(lo_radial_solutions_[i].size(), 1, &a[0][0], 4, b, 4);

                if (info) 
                {
                    std::stringstream s;
                    s << "gesv returned " << info;
                    error(__FILE__, __LINE__, s);
                }
                
                // take linear combination of radial solutions
                for (int order = 0; order < (int)lo_radial_solutions_[i].size(); order++)
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        lo_radial_functions_(ir, i, 0) += b[order] * lo_radial_solutions_[i][order].p[ir];
                        lo_radial_functions_(ir, i, 1) += b[order] * lo_radial_solutions_[i][order].hp[ir];
                    }

                // normalize
                for (int ir = 0; ir < nmtp; ir++)
                    s[ir] = lo_radial_functions_(ir, i, 0) * lo_radial_functions_(ir, i, 0);
                s.interpolate();
                double norm = s.integrate(2);
                norm = 1.0 / sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++)
                {
                    lo_radial_functions_(ir, i, 0) *= norm;
                    lo_radial_functions_(ir, i, 1) *= norm;
                }

                /*std::ofstream out("lo.dat");
                for (int ir = 0; ir < nmtp; ir++)
                   out << atom_type_->radial_grid()[ir] << " " << lo_radial_functions_(ir, i, 0) << std::endl;
                out.close();*/

                if (fabs(lo_radial_functions_(nmtp - 1, i, 0)) > 1e-10)
                {
                    std::stringstream s;
                    s << "local orbital is not zero at MT boundary" << std::endl 
                      << "  value : " << lo_radial_functions_(nmtp - 1, i, 0);
                    error(__FILE__, __LINE__, s);
                }
            }
        }

        void generate_radial_functions()
        {
            generate_aw_radial_functions();
            generate_lo_radial_functions();

            for (int i = 0; i < atom_type_->indexr().size(); i++)
            {
                int l =  atom_type_->indexr(i).l();
                int order =  atom_type_->indexr(i).order();
                int idxlo = atom_type_->indexr(i).idxlo();
                if (idxlo < 0) 
                {
                    memcpy(&radial_functions_(0, i, 0), &aw_radial_functions_(0, order, l, 0), atom_type_->num_mt_points() * sizeof(double));
                    memcpy(&radial_functions_(0, i, 1), &aw_radial_functions_(0, order, l, 1), atom_type_->num_mt_points() * sizeof(double));
                }
                else
                {
                    memcpy(&radial_functions_(0, i, 0), &lo_radial_functions_(0, idxlo, 0), atom_type_->num_mt_points() * sizeof(double));
                    memcpy(&radial_functions_(0, i, 1), &lo_radial_functions_(0, idxlo, 1), atom_type_->num_mt_points() * sizeof(double));
                }
            }

            print_enu();
        }

        void print_enu()
        {
            printf("augmented waves\n");
            for (int i = 0; i < (int)aw_radial_solutions_.size(); i++)
                for (int order = 0; order < (int)aw_radial_solutions_[i].size(); order++)
                    printf("n = %i   l = %i   order = %i   enu = %f\n", aw_radial_solutions_[i][order].rsd.n, aw_radial_solutions_[i][order].rsd.l,
                                                                        order, aw_radial_solutions_[i][order].rsd.enu);
            printf("local orbitals\n");
            for (int i = 0; i < (int)lo_radial_solutions_.size(); i++)
                for (int order = 0; order < (int)lo_radial_solutions_[i].size(); order++)
                    printf("n = %i   l = %i   order = %i   enu = %f\n", lo_radial_solutions_[i][order].rsd.n, lo_radial_solutions_[i][order].rsd.l,
                                                                        order, lo_radial_solutions_[i][order].rsd.enu);
         }
};

};

#endif // __ATOM_SYMMETRY_CLASS_H__

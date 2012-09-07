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

        mdarray<double,3> radial_integrals;
        
        std::vector<int> atom_id_;
        
        AtomType* atom_type_;

        std::vector<double> spherical_potential_;

        std::vector<radial_solution_set> aw_radial_solutions_;
        
        std::vector<radial_solution_set> lo_radial_solutions_;



        mdarray<double,3> lo_radial_functions_;

        //mdarray<double,2> radial_functions_;
        
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

            lo_radial_functions_.set_dimensions(atom_type_->num_mt_points(), lo_radial_solutions_.size(), 2);
            lo_radial_functions_.allocate();
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

        void generate_radial_functions()
        {
            RadialSolver solver(false, -1.0 * atom_type_->zn(), atom_type_->radial_grid());

            std::vector<double> p;

            for (int i = 0; i < (int)aw_radial_solutions_.size(); i++)
                for (int order = 0; order < (int)aw_radial_solutions_[i].size(); order++)
                {
                    // find linearization energies
                    if (aw_radial_solutions_[i][order].rsd.auto_enu)
                        solver.bound_state(aw_radial_solutions_[i][order].rsd.n, aw_radial_solutions_[i][order].rsd.l, spherical_potential_, 
                                           aw_radial_solutions_[i][order].rsd.enu, p);

                    solver.solve_in_mt(aw_radial_solutions_[i][order].rsd.l, aw_radial_solutions_[i][order].rsd.enu, 
                                       aw_radial_solutions_[i][order].rsd.dme, spherical_potential_, 
                                       aw_radial_solutions_[i][order].p, aw_radial_solutions_[i][order].hp);
                }

            for (int i = 0; i < (int)lo_radial_solutions_.size(); i++)
                for (int order = 0; order < (int)lo_radial_solutions_[i].size(); order++)
                {
                    // find linearization energies
                    if (lo_radial_solutions_[i][order].rsd.auto_enu)
                        solver.bound_state(lo_radial_solutions_[i][order].rsd.n, lo_radial_solutions_[i][order].rsd.l, spherical_potential_, 
                                           lo_radial_solutions_[i][order].rsd.enu, p);

                    solver.solve_in_mt(lo_radial_solutions_[i][order].rsd.l, lo_radial_solutions_[i][order].rsd.enu, 
                                       lo_radial_solutions_[i][order].rsd.dme, spherical_potential_, 
                                       lo_radial_solutions_[i][order].p, lo_radial_solutions_[i][order].hp);
                }

            // generate local orbitals
            int nmtp = atom_type_->num_mt_points();
            Spline<double> s(nmtp, atom_type_->radial_grid());
            double a[4][4];

            for (int i = 0; i < (int)lo_radial_solutions_.size(); i++)
            {
                assert(lo_radial_solutions_[i].size() < 4);

                for (int order = 0; order < (int)lo_radial_solutions_[i].size(); order++)
                {
                    // normalize radial solutions
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = lo_radial_solutions_[i][order].p[ir] * lo_radial_solutions_[i][order].p[ir];
                    s.interpolate();
                    double norm = s.integrate(0);
                    norm = 1.0 / sqrt(norm);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        lo_radial_solutions_[i][order].p[ir] *= norm;
                        //lo_radial_solutions_[i][order].hp[ir] *= norm;
                    }

                    // compute radial derivatives
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = lo_radial_solutions_[i][order].p[ir] / atom_type_->radial_grid()[ir];
                    s.interpolate();

                    for (int dm = 0; dm < (int)lo_radial_solutions_[i].size(); dm++)
                        a[order][dm] = s.deriv(dm, nmtp - 1, atom_type_->radial_grid().dr(nmtp - 1));
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

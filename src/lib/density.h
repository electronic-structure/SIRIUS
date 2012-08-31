namespace sirius
{

class Density
{
    //private:
    public:
        mdarray<double,3> charge_density_mt_;
        mdarray<double,1> charge_density_it_;
        mdarray<complex16,1> charge_density_pw_;
        
        mdarray<double,3> magnetization_density_mt_;
        mdarray<double,2> magnetization_density_it_;

        PeriodicFunction<double> charge_density_;
        
    
        
    public:
    
        void initialize()
        {
            charge_density_mt_.set_dimensions(global.lmmax_rho(), global.max_num_mt_points(), global.num_atoms());
            charge_density_mt_.allocate();
            charge_density_it_.set_dimensions(global.fft().size());
            charge_density_it_.allocate();
            charge_density_pw_.set_dimensions(global.num_gvec());
            charge_density_pw_.allocate();
            
            charge_density_.allocate(global.lmax_rho(), global.max_num_mt_points(), global.num_atoms(), 
                                     global.fft().size(), global.num_gvec());
        }
        
        void get_density(double* _rhomt, double* _rhoir)
        {
            memcpy(_rhomt, charge_density_mt_.get_ptr(), charge_density_mt_.size() * sizeof(double)); 
            memcpy(_rhoir, charge_density_it_.get_ptr(), charge_density_it_.size() * sizeof(double)); 
        }
    
        void initial_density()
        {
            std::vector<double> enu;
            for (int i = 0; i < global.num_atom_types(); i++)
                global.atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

            charge_density_.zero();

            charge_density_mt_.zero();
            double mt_charge = 0.0;
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int nmtp = global.atom(ia)->type()->num_mt_points();
                Spline rho(nmtp, global.atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++)
                {
                    rho[ir] = global.atom(ia)->type()->free_atom_density(ir);
                    charge_density_mt_(0, ir, ia) = rho[ir] / y00; 
                    charge_density_mt_(1, ir, ia) = rho[ir] / y00; // TODO: remove later after tests
                    charge_density_.frlm(0, ir, ia) = rho[ir] / y00;
                    charge_density_.frlm(1, ir, ia) = rho[ir] / y00;
                }
                rho.interpolate();
                mt_charge += fourpi * rho.integrate(nmtp - 1, 2);
            }
            
            for (int i = 0; i < global.fft().size(); i++)
                charge_density_it_(i) = (global.num_electrons() - mt_charge) / global.volume_it();
            
            global.fft().transform(&charge_density_it_(0), NULL);

            complex16* fft_buf = global.fft().output_buffer_ptr();
            for (int ig = 0; ig < global.num_gvec(); ig++)
                charge_density_pw_(ig) = fft_buf[global.fft_index(ig)];
        }



};

Density density;

};

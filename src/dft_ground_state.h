
namespace sirius
{

class DFT_ground_state
{
    Global* parameters_;

    Potential* potential_;

    Density* density_;

    kset* kset_;

    public:

        DFT_ground_state(Global* parameters__, Potential* potential__, Density* density__, kset* kset__) : 
            parameters_(parameters__), potential_(potential__), density_(density__), kset_(kset__)
        {

        }

        void forces()
        {
            mdarray<double, 2> atom_force(3, parameters_->num_atoms());

            kset_->force(atom_force);

            mt_function<double>* g[3];
            for (int x = 0; x < 3; x++) 
            {
                g[x] = new mt_function<double>(Argument(arg_lm, parameters_->lmmax_pot()), 
                                               Argument(arg_radial, parameters_->max_num_mt_points()));
            }
            
            for (int ialoc = 0; ialoc < parameters_->spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_->spl_num_atoms(ialoc);
                gradient(parameters_->atom(ia)->type()->radial_grid(), potential_->coulomb_potential_mt(ialoc), g[0], g[1], g[2]);
                for (int x = 0; x < 3; x++) atom_force(x, ia) += parameters_->atom(ia)->type()->zn() * (*g[x])(0, 0) * y00;
            }
            
            for (int x = 0; x < 3; x++) 
            {
                delete g[x];
                g[x] = new mt_function<double>(Argument(arg_lm, parameters_->lmmax_rho()), 
                                               Argument(arg_radial, parameters_->max_num_mt_points()));
            }

            for (int ialoc = 0; ialoc < parameters_->spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_->spl_num_atoms(ialoc);
                gradient(parameters_->atom(ia)->type()->radial_grid(), density_->density_mt(ialoc), g[0], g[1], g[2]);
                for (int x = 0; x < 3; x++)
                {
                    atom_force(x, ia) += inner(parameters_->atom(ia)->type()->radial_grid(), 
                                               potential_->effective_potential_mt(ialoc), g[x]);
                }
            }

            for (int x = 0; x < 3; x++) delete g[x];



        }


};





















};

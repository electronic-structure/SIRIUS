namespace sirius
{

class Density
{
    public:
    
        void initial_density()
        {
            std::vector<double> enu;
            for (int i = 0; i < sirius_global.num_atom_types(); i++)
                sirius_global.atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);
        }


};


};

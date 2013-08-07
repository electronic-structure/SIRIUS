
namespace sirius
{

class DFT_ground_state
{
    private:

        Global* parameters_;

        Potential* potential_;

        Density* density_;

        kset* kset_;

    public:

        DFT_ground_state(Global* parameters__, Potential* potential__, Density* density__, kset* kset__) : 
            parameters_(parameters__), potential_(potential__), density_(density__), kset_(kset__)
        {

        }

        void move_atoms(int istep);

        void forces(mdarray<double, 2>& atom_force);

        void scf_loop();

        void relax_atom_positions();
};

#include "dft_ground_state.hpp"

};


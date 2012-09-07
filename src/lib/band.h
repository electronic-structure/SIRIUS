
namespace sirius
{

class Band
{
    public:

        void radial()
        {
            // save spherical part of potential
            for (int ic = 0; ic < global.num_atom_symmetry_classes(); ic++)
            {
               int ia = global.atom_symmetry_class(ic)->atom_id(0);
               int nmtp = global.atom(ia)->type()->num_mt_points();
               
               std::vector<double> veff(nmtp);
               
               for (int ir = 0; ir < nmtp; ir++)
                   veff[ir] = y00 * potential.effective_potential().f_rlm(0, ir, ia);

               global.atom_symmetry_class(ic)->set_spherical_potential(veff);
            }

            find_enu();

        }

        void find_enu()
        {
            for (int ic = 0; ic < global.num_atom_symmetry_classes(); ic++)
                global.atom_symmetry_class(ic)->generate_radial_functions();


        }

};

Band band;

};

#include <sirius.hpp>
#include "testing.hpp"

using namespace sirius;

int
enu_from_potential(cmd_args const& args__)
{
    auto species_file = args__.value<std::string>("species");
    auto pot_file     = args__.value<std::string>("potential");
    auto rel          = get_relativity_t(args__.value<std::string>("rel", "none"));

    Simulation_parameters params;
    params.electronic_structure_method("full_potential_lapwlo");
    params.valence_relativity(args__.value<std::string>("rel", "none"));
    params.verbosity(4);

    std::ifstream ifs(pot_file);
    nlohmann::json dict;
    ifs >> dict;

    auto x    = dict["x"].get<std::vector<double>>();
    auto veff = dict["veff"].get<std::vector<double>>();
    auto z    = dict["z"].get<double>();

    Atom_type atype(params, 0, "X", species_file);
    atype.init();
    atype.set_radial_grid(x.size(), x.data());

    if (static_cast<int>(x.size()) != atype.num_mt_points() ||
        static_cast<int>(veff.size()) != atype.num_mt_points()) {
        std::stringstream s;
        s << "radial grid or potential size does not match species file" << std::endl
          << "  x.size              : " << x.size() << std::endl
          << "  atype.num_mt_points : " << atype.num_mt_points() << std::endl
          << "  veff.size           : " << veff.size();
        RTE_THROW(s.str());
    }

    if (std::abs(z - atype.zn()) > 1e-12) {
        RTE_WARNING("nuclear charge in potential dump differs from species file");
    }

    for (int ir = 0; ir < atype.num_mt_points(); ir++) {
        if (std::abs(x[ir] - atype.radial_grid(ir)) > 1e-10) {
            std::stringstream s;
            s << "radial grid in potential dump does not match species file" << std::endl
              << "  ir                    : " << ir << std::endl
              << "  x[ir]                 : " << x[ir] << std::endl
              << "  atype.radial_grid(ir) : " << atype.radial_grid(ir) << std::endl
              << "  mt_radius             : " << atype.mt_radius();
            RTE_THROW(s.str());
        }
    }

    Atom_symmetry_class atom_class(0, atype);
    atom_class.set_spherical_potential(veff);

    auto ierr = atom_class.find_enu(rel);

    atom_class.write_enu(std::cout);

    ierr += atom_class.generate_radial_functions(rel, false);

    return ierr;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"species=", "(string) species file"},
                   {"potential=", "(string) spherical potential JSON"},
                   {"rel=", "(string) valence relativity"}});

    sirius::initialize(true);
    int result = call_test("enu_from_potential", enu_from_potential, args);
    sirius::finalize();
    return result;
}

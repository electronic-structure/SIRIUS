#include "apps.hpp"
//#include "k_point/generate_w90_coeffs.hpp"
//#include "nlcglib/apply_hamiltonian.hpp"
#include "hamiltonian/check_wave_functions.hpp"

std::vector<std::array<double, 3>>
load_coordinates(const std::string& fname__)
{
    std::vector<std::array<double, 3>> kp;

    //read k coordinates from hdf5
    HDF5_tree fin(fname__, hdf5_access_t::read_only);
    std::cout << "read num_kpoints" << std::endl;
    int num_kpoints;
    fin["K_point_set"].read("num_kpoints", &num_kpoints, 1);
    std::cout << "num_kpoints: " << num_kpoints << std::endl;
    kp.resize(num_kpoints);
    for (int ik = 0; ik < num_kpoints; ik++) {
        fin["K_point_set"][ik].read("vk", &kp[ik][0], 3);
        std::cout << "ik = " << ik << " kp = { " << kp[ik][0] << " " << kp[ik][1] << " " << kp[ik][2] << std::endl;
    }
    return kp;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"input=", "{string} input file name"}});

    sirius::initialize(1);

    /* get the input file name */
    auto fpath = args.value<fs::path>("input", "state.h5");

    if (fs::is_directory(fpath)) {
        fpath /= "sirius.h5";
    }

    if (!fs::exists(fpath)) {
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "input file does not exist" << std::endl;
        }
        exit(1);
    }
    auto fname = fpath.string();

    /* create simulation context */
    auto ctx = create_sim_ctx(fname, args);
    ctx->initialize();

    /* read the wf */
    auto kp = load_coordinates(fname);
    K_point_set kset(*ctx, kp);
    std::cout << "kset initialized.\n";
    kset.load(fname);

    /* initialize the ground state */
    DFT_ground_state dft(kset);
    auto& potential = dft.potential();
    auto& density   = dft.density();
    density.load(fname);
    density.generate_paw_density();
    //potential.load(fname);
    potential.generate(density, ctx->use_symmetry(), true);
    Hamiltonian0<double> H0(potential, true);

    /* checksum over wavefunctions */
    //for (auto it : kset.spl_num_kpoints()) {
    //    int ik = it.i;
    //    auto Hk = H0(*kset.get<double>(ik));
    //    for (auto is=0; is< ctx->num_spins(); is++) {
    //      std::cout << "ik: " << ik << " ispn : "<< is << " " ;
    //      std::cout << kset.get<double>(ik)->spinor_wave_functions().checksum(memory_t::host, wf::spin_index(is), wf::band_range(0, ctx->num_bands())) << std::endl;
    //    }
    //}
    /* check if the wfs diagonalize the hamiltonian and if the eigenvalues are correct */

    std::cout << "num_spins " << kset.ctx().num_spins() << std::endl;

    for (auto it : kset.spl_num_kpoints()) {
        int ik = it.i;
        std::cout << "ik = " << ik << std::endl;
        auto kp = kset.get<double>(ik);
        auto Hk = H0(*kp);

        /* check wave-functions */
        if (true || ctx->cfg().control().verification() >= 2) {
            if (ctx->num_mag_dims() == 3) {
                auto eval = kp->band_energies(0);
                check_wave_functions<double, std::complex<double>>(Hk, kp->spinor_wave_functions(),
                                                                   wf::spin_range(0, 2),
                                                                   wf::band_range(0, ctx->num_bands()), eval.data());
            } else {
                for (int ispn = 0; ispn < ctx->num_spins(); ispn++) {
                    auto eval = kp->band_energies(ispn);
                    check_wave_functions<double, std::complex<double>>(
                            Hk, kp->spinor_wave_functions(), wf::spin_range(ispn), wf::band_range(0, ctx->num_bands()),
                            eval.data());
                }
            }
        }

    } //kpoint

    //kset.generate_w90_coeffs();
}

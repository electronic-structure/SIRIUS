#include <gsl/gsl_sf_bessel.h>
#include "sirius_version.hpp"
#include "simulation_context.hpp"
#include "Symmetry/find_lat_sym.hpp"
#include "utils/profiler.hpp"

namespace sirius {

void print_memory_usage(const char* file__, int line__)
{
    size_t VmRSS, VmHWM;
    utils::get_proc_status(&VmHWM, &VmRSS);

    std::vector<char> str(2048);

    int n = snprintf(&str[0], 2048, "[rank%04i at line %i of file %s]", Communicator::world().rank(), line__, file__);

    n += snprintf(&str[n], 2048, " VmHWM: %i Mb, VmRSS: %i Mb", static_cast<int>(VmHWM >> 20),
                  static_cast<int>(VmRSS >> 20));

    if (acc::num_devices() > 0) {
        size_t gpu_mem = acc::get_free_mem();
        n += snprintf(&str[n], 2048, ", GPU free memory: %i Mb", static_cast<int>(gpu_mem >> 20));
    }
    std::printf("%s\n", &str[0]);
}

double unit_step_function_form_factors(double R__, double g__)
{
    if (g__ < 1e-12) {
        return std::pow(R__, 3) / 3.0;
    } else {
        return (std::sin(g__ * R__) - g__ * R__ * std::cos(g__ * R__)) / std::pow(g__, 3);
    }
}

void Simulation_context::init_fft_grid()
{
    if (!(control().fft_mode_ == "serial" || control().fft_mode_ == "parallel")) {
        TERMINATE("wrong FFT mode");
    }

    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    /* create FFT driver for dense mesh (density and potential) */
    auto fft_grid = settings().fft_grid_size_;
    if (fft_grid[0] * fft_grid[1] * fft_grid[2] == 0) {
        fft_grid_ = get_min_fft_grid(pw_cutoff(), rlv);
    } else {
        /* else create a grid with user-specified dimensions */
        fft_grid_ = sddk::FFT3D_grid(fft_grid);
    }

    /* create FFT grid for coarse mesh */
    fft_coarse_grid_ = get_min_fft_grid(2 * gk_cutoff(), rlv);
}

mdarray<double, 3> Simulation_context::generate_sbessel_mt(int lmax__) const
{
    PROFILE("sirius::Simulation_context::generate_sbessel_mt");

    mdarray<double, 3> sbessel_mt(lmax__ + 1, gvec().count(), unit_cell().num_atom_types());
    for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            auto gv = gvec().gvec_cart<index_domain_t::local>(igloc);
            gsl_sf_bessel_jl_array(lmax__, gv.length() * unit_cell().atom_type(iat).mt_radius(),
                                   &sbessel_mt(0, igloc, iat));
        }
    }
    return sbessel_mt;
}

matrix<double_complex> Simulation_context::generate_gvec_ylm(int lmax__)
{
    PROFILE("sirius::Simulation_context::generate_gvec_ylm");

    matrix<double_complex> gvec_ylm(utils::lmmax(lmax__), gvec().count(), memory_t::host, "gvec_ylm");
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        auto rtp = SHT::spherical_coordinates(gvec().gvec_cart<index_domain_t::local>(igloc));
        sf::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, igloc));
    }
    return gvec_ylm;
}

mdarray<double_complex, 2>
Simulation_context::sum_fg_fl_yg(int lmax__, double_complex const* fpw__, mdarray<double, 3>& fl__,
                                 matrix<double_complex>& gvec_ylm__)
{
    PROFILE("sirius::Simulation_context::sum_fg_fl_yg");

    int ngv_loc = gvec().count();

    int na_max{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        na_max = std::max(na_max, unit_cell_.atom_type(iat).num_atoms());
    }

    int lmmax = utils::lmmax(lmax__);
    /* resuling matrix */
    mdarray<double_complex, 2> flm(lmmax, unit_cell().num_atoms());

    matrix<double_complex> phase_factors;
    matrix<double_complex> zm;
    matrix<double_complex> tmp;

    switch (processing_unit()) {
        case device_t::CPU: {
            auto& mp      = mem_pool(memory_t::host);
            phase_factors = matrix<double_complex>(ngv_loc, na_max, mp);
            zm            = matrix<double_complex>(lmmax, ngv_loc, mp);
            tmp           = matrix<double_complex>(lmmax, na_max, mp);
            break;
        }
        case device_t::GPU: {
            auto& mp      = mem_pool(memory_t::host);
            auto& mpd     = mem_pool(memory_t::device);
            phase_factors = matrix<double_complex>(nullptr, ngv_loc, na_max);
            phase_factors.allocate(mpd);
            zm = matrix<double_complex>(lmmax, ngv_loc, mp);
            zm.allocate(mpd);
            tmp = matrix<double_complex>(lmmax, na_max, mp);
            tmp.allocate(mpd);
            break;
        }
    }

    std::vector<double_complex> zil(lmax__ + 1);
    for (int l = 0; l <= lmax__; l++) {
        zil[l] = std::pow(double_complex(0, 1), l);
    }

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        int na = unit_cell_.atom_type(iat).num_atoms();
        generate_phase_factors(iat, phase_factors);
        PROFILE_START("sirius::Simulation_context::sum_fg_fl_yg|zm");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ngv_loc; igloc++) {
            for (int l = 0, lm = 0; l <= lmax__; l++) {
                double_complex z = fourpi * fl__(l, igloc, iat) * zil[l] * fpw__[igloc];
                for (int m = -l; m <= l; m++, lm++) {
                    zm(lm, igloc) = z * std::conj(gvec_ylm__(lm, igloc));
                }
            }
        }
        PROFILE_STOP("sirius::Simulation_context::sum_fg_fl_yg|zm");
        PROFILE_START("sirius::Simulation_context::sum_fg_fl_yg|mul");
        switch (processing_unit()) {
            case device_t::CPU: {
                linalg(linalg_t::blas).gemm('N', 'N', lmmax, na, ngv_loc,
                    &linalg_const<double_complex>::one(), zm.at(memory_t::host), zm.ld(),
                    phase_factors.at(memory_t::host), phase_factors.ld(),
                    &linalg_const<double_complex>::zero(), tmp.at(memory_t::host), tmp.ld());
                break;
            }
            case device_t::GPU: {
                zm.copy_to(memory_t::device);
                linalg(linalg_t::gpublas).gemm('N', 'N', lmmax, na, ngv_loc, 
                    &linalg_const<double_complex>::one(), zm.at(memory_t::device), zm.ld(),
                    phase_factors.at(memory_t::device), phase_factors.ld(),
                    &linalg_const<double_complex>::zero(), tmp.at(memory_t::device), tmp.ld());
                tmp.copy_to(memory_t::host);
                break;
            }
        }
        PROFILE_STOP("sirius::Simulation_context::sum_fg_fl_yg|mul");

        for (int i = 0; i < na; i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            std::copy(&tmp(0, i), &tmp(0, i) + lmmax, &flm(0, ia));
        }
    }

    comm().allreduce(&flm(0, 0), (int)flm.size());

    return flm;
}

double Simulation_context::ewald_lambda() const
{
    /* alpha = 1 / (2*sigma^2), selecting alpha here for better convergence */
    double lambda{1};
    double gmax = pw_cutoff();
    double upper_bound{0};
    double charge = unit_cell_.num_electrons();

    /* iterate to find lambda */
    do {
        lambda += 0.1;
        upper_bound =
            charge * charge * std::sqrt(2.0 * lambda / twopi) * std::erfc(gmax * std::sqrt(1.0 / (4.0 * lambda)));
    } while (upper_bound < 1e-8);

    if (lambda < 1.5 && comm().rank() == 0) {
        std::stringstream s;
        s << "ewald_lambda(): pw_cutoff is too small";
        WARNING(s);
    }
    return lambda;
}

splindex<splindex_t::block> Simulation_context::split_gvec_local() const
{
    /* local number of G-vectors for this MPI rank */
    int ngv_loc = gvec().count();
    /* estimate number of G-vectors in a block */
    int ld{-1};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        int nat = unit_cell_.atom_type(iat).num_atoms();
        int nbf = unit_cell_.atom_type(iat).mt_basis_size();

        ld = std::max(ld, std::max(nbf * (nbf + 1) / 2, nat));
    }
    /* limit the size of relevant array to ~1Gb */
    int ngv_b = (1 << 30) / sizeof(double_complex) / ld;
    ngv_b = std::max(1, std::min(ngv_loc, ngv_b));
    /* number of blocks of G-vectors */
    int nb = ngv_loc / ngv_b;
    /* split local number of G-vectors between blocks */
    return splindex<splindex_t::block>(ngv_loc, nb, 0);
}

void Simulation_context::initialize()
{
    PROFILE("sirius::Simulation_context::initialize");

    /* can't initialize twice */
    if (initialized_) {
        TERMINATE("Simulation parameters are already initialized.");
    }
    electronic_structure_method(parameters_input().electronic_structure_method_);
    set_core_relativity(parameters_input().core_relativity_);
    set_valence_relativity(parameters_input().valence_relativity_);

    /* can't run fp-lapw with Gamma point trick */
    if (full_potential()) {
        gamma_point(false);
    }

    /* Gamma-point calculation and non-collinear magnetism are not compatible */
    if (num_mag_dims() == 3) {
        gamma_point(false);
    }

    /* set processing unit type */
    set_processing_unit(control().processing_unit_);

    /* check if we can use a GPU device */
    if (processing_unit() == device_t::GPU) {
#if !defined(__GPU)
        throw std::runtime_error("not compiled with GPU support!");
#endif
    }

    /* initialize MPI communicators */
    init_comm();

    if (control().verbosity_ >= 3) {
        pstdout pout(comm());
        if (comm().rank() == 0) {
            pout.printf("MPI rank placement\n");
            pout.printf("------------------\n");
        }
        pout.printf("rank: %3i, comm_band_rank: %3i, comm_k_rank: %3i, hostname: %s\n", comm().rank(),
                    comm_band().rank(), comm_k().rank(), utils::hostname().c_str());
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            host_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            host_memory_t_ = memory_t::host_pinned;
            break;
        }
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            preferred_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            if (control_input_.memory_usage_ == "high") {
                preferred_memory_t_ = memory_t::device;
            }
            if (control_input_.memory_usage_ == "low" || control_input_.memory_usage_ == "medium") {
                preferred_memory_t_ = memory_t::host_pinned;
            }
            break;
        }
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            aux_preferred_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            if (control_input_.memory_usage_ == "high" || control_input_.memory_usage_ == "medium") {
                aux_preferred_memory_t_ = memory_t::device;
            }
            if (control_input_.memory_usage_ == "low") {
                aux_preferred_memory_t_ = memory_t::host_pinned;
            }
            break;
        }
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            blas_linalg_t_ = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            if (control_input_.memory_usage_ == "high") {
                blas_linalg_t_ = linalg_t::gpublas;
            }
            if (control_input_.memory_usage_ == "low" || control_input_.memory_usage_ == "medium") {
#ifdef __ROCM
                blas_linalg_t_ = linalg_t::gpublas;
#else
                blas_linalg_t_ = linalg_t::cublasxt;
#endif
            }
            break;
        }
    }

    /* can't use reduced G-vectors in LAPW code */
    if (full_potential()) {
        control_input_.reduce_gvec_ = false;
    }

    if (!iterative_solver_input_.type_.size()) {
        if (full_potential()) {
            iterative_solver_input_.type_ = "exact";
        } else {
            iterative_solver_input_.type_ = "davidson";
        }
    }
    /* set default values for the G-vector cutoff */
    if (pw_cutoff() <= 0) {
        pw_cutoff(full_potential() ? 12 : 20);
    }

    /* initialize variables related to the unit cell */
    unit_cell_.initialize();
    /* save the volume of the initial unit cell */
    omega0_ = unit_cell_.omega();

    /* check the lattice symmetries */
    if (use_symmetry()) {
        auto lv = matrix3d<double>(unit_cell_.lattice_vectors());

        auto lat_sym = find_lat_sym(lv, 1e-6);

        #pragma omp parallel for
        for (int i = 0; i < unit_cell_.symmetry().num_mag_sym(); i++) {
            auto& spgR = unit_cell_.symmetry().magnetic_group_symmetry(i).spg_op.R;
            bool found{false};
            for (size_t i = 0; i < lat_sym.size(); i++) {
                auto latR = lat_sym[i];
                found     = true;
                for (int x : {0, 1, 2}) {
                    for (int y : {0, 1, 2}) {
                        found = found && (spgR(x, y) == latR(x, y));
                    }
                }
                if (found) {
                    break;
                }
            }
            if (!found) {
                TERMINATE("spglib lattice symetry was not found in the list of SIRIUS generated symmetries");
            }
        }
    }

    /* set default values for the G+k-vector cutoff */
    if (full_potential()) {
        /* find the cutoff for G+k vectors (derived from rgkmax (aw_cutoff here) and minimum MT radius) */
        if (aw_cutoff() > 0) {
            gk_cutoff(aw_cutoff() / unit_cell_.min_mt_radius());
        }
        if (gk_cutoff() <= 0) {
            gk_cutoff(3);
        }
    } else {
        /* in pseudopotential case */
        if (gk_cutoff() <= 0) {
            gk_cutoff(7);
        }
    }

    /* check the G+k cutoff */
    if (gk_cutoff() * 2 > pw_cutoff()) {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff() * 2;
        TERMINATE(s);
    }

    if (!full_potential()) {
        set_lmax_rho(unit_cell_.lmax() * 2);
        set_lmax_pot(unit_cell_.lmax() * 2);
        set_lmax_apw(-1);
    }

    /* initialize FFT grid dimensions */
    init_fft_grid();

    int nbnd = static_cast<int>(unit_cell_.num_valence_electrons() / 2.0) +
               std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons()));
    if (full_potential()) {
        /* take 10% of empty non-magnetic states */
        if (num_fv_states() < 0) {
            num_fv_states(nbnd);
        }
        if (num_fv_states() < static_cast<int>(unit_cell_.num_valence_electrons() / 2.0)) {
            std::stringstream s;
            s << "not enough first-variational states : " << num_fv_states();
            TERMINATE(s);
        }
    } else {
        if (num_mag_dims() == 3) {
            nbnd *= 2;
        }
        if (num_bands() < 0) {
            num_bands(nbnd);
        }
    }

    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};
#if defined(__MAGMA)
    bool is_magma{true};
#else
    bool is_magma{false};
#endif
#if defined(__SCALAPACK)
    bool is_scalapack{true};
#else
    bool is_scalapack{false};
#endif
#if defined(__ELPA)
    bool is_elpa{true};
#else
    bool is_elpa{false};
#endif

    int npr = control_input_.mpi_grid_dims_[0];
    int npc = control_input_.mpi_grid_dims_[1];

    /* deduce the default eigen-value solver */
    for (int i : {0, 1}) {
        if (evsn[i] == "") {
            /* conditions for sequential diagonalization */
            if (comm_band().size() == 1 || npc == 1 || npr == 1 || !is_scalapack) {
                if (is_magma && num_bands() > 200) {
                    evsn[i] = "magma";
                } else {
                    evsn[i] = "lapack";
                }
            } else {
                if (is_scalapack) {
                    evsn[i] = "scalapack";
                }
                if (is_elpa) {
                    evsn[i] = "elpa1";
                }
            }
        }
    }

    std_evp_solver_name(evsn[0]);
    gen_evp_solver_name(evsn[1]);

    std_evp_solver_ = Eigensolver_factory(std_evp_solver_type());
    gen_evp_solver_ = Eigensolver_factory(gen_evp_solver_type());

    auto& std_solver = std_evp_solver();
    auto& gen_solver = gen_evp_solver();

    if (std_solver.is_parallel() != gen_solver.is_parallel()) {
        TERMINATE("both solvers must be sequential or parallel");
    }

    /* setup BLACS grid */
    if (std_solver.is_parallel()) {
        blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(comm_band(), npr, npc));
    } else {
        blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), 1, 1));
    }

    /* setup the cyclic block size */
    if (cyclic_block_size() < 0) {
        double a = std::min(std::log2(double(num_bands()) / blacs_grid_->num_ranks_col()),
                            std::log2(double(num_bands()) / blacs_grid_->num_ranks_row()));
        if (a < 1) {
            control_input_.cyclic_block_size_ = 2;
        } else {
            control_input_.cyclic_block_size_ =
                static_cast<int>(std::min(128.0, std::pow(2.0, static_cast<int>(a))) + 1e-12);
        }
    }

    /* placeholder for augmentation operator for each atom type */
    augmentation_op_.resize(unit_cell().num_atom_types());

    /* create G-vectors on the first call to update() */
    update();

    this->print_memory_usage(__FILE__, __LINE__);

    if (control().verbosity_ >= 1 && comm().rank() == 0) {
        print_info();
    }

    initialized_ = true;
}

void Simulation_context::print_info() const
{
    tm const* ptm = localtime(&start_time_.tv_sec);
    char buf[100];
    strftime(buf, sizeof(buf), "%a, %e %b %Y %H:%M:%S", ptm);

    std::printf("\n");
    std::printf("SIRIUS version : %i.%i.%i\n", sirius::major_version(), sirius::minor_version(), sirius::revision());
    std::printf("git hash       : %s\n", sirius::git_hash().c_str());
    std::printf("git branch     : %s\n", sirius::git_branchname().c_str());
    std::printf("build time     : %s\n", sirius::build_date().c_str());
    std::printf("start time     : %s\n", buf);
    std::printf("\n");
    std::printf("number of MPI ranks           : %i\n", comm_.size());
    if (mpi_grid_) {
        std::printf("MPI grid                      :");
        for (int i = 0; i < mpi_grid_->num_dimensions(); i++) {
            std::printf(" %i", mpi_grid_->communicator(1 << i).size());
        }
        std::printf("\n");
    }
    std::printf("maximum number of OMP threads : %i\n", omp_get_max_threads());
    std::printf("number of MPI ranks per node  : %i\n", num_ranks_per_node());
    std::printf("page size (Kb)                : %li\n", utils::get_page_size() >> 10);
    std::printf("number of pages               : %li\n", utils::get_num_pages());
    std::printf("available memory (GB)         : %li\n", utils::get_total_memory() >> 30);

    std::string headers[]         = {"FFT context for density and potential", "FFT context for coarse grid"};
    double cutoffs[]              = {pw_cutoff(), 2 * gk_cutoff()};
    Communicator const* comms[]   = {&comm_fft(), &comm_fft_coarse()};
    FFT3D_grid fft_grids[]        = {this->fft_grid_, this->fft_coarse_grid_};
    Gvec const* gvecs[]           = {&gvec(), &gvec_coarse()};

    std::printf("\n");
    for (int i = 0; i < 2; i++) {
        std::printf("%s\n", headers[i].c_str());
        std::printf("=====================================\n");
        std::printf("  comm size                             : %i\n", comms[i]->size());
        std::printf("  plane wave cutoff                     : %f\n", cutoffs[i]);
        std::printf("  grid size                             : %i %i %i   total : %i\n", fft_grids[i][0],
               fft_grids[i][1], fft_grids[i][2], fft_grids[i].num_points());
        std::printf("  grid limits                           : %i %i   %i %i   %i %i\n", fft_grids[i].limits(0).first,
               fft_grids[i].limits(0).second, fft_grids[i].limits(1).first, fft_grids[i].limits(1).second,
               fft_grids[i].limits(2).first, fft_grids[i].limits(2).second);
        std::printf("  number of G-vectors within the cutoff : %i\n", gvecs[i]->num_gvec());
        std::printf("  local number of G-vectors             : %i\n", gvecs[i]->count());
        std::printf("  number of G-shells                    : %i\n", gvecs[i]->num_shells());
        std::printf("\n");
    }
    std::printf("number of local G-vector blocks: %i\n", split_gvec_local().num_ranks());

    unit_cell_.print_info(control().verbosity_);
    for (int i = 0; i < unit_cell_.num_atom_types(); i++) {
        unit_cell_.atom_type(i).print_info();
    }

    std::printf("\n");
    std::printf("Basic information\n");
    for (int i = 0; i < 80; i++, std::printf("-"));
    std::printf("\n");
    std::printf("total nuclear charge               : %i\n", unit_cell().total_nuclear_charge());
    std::printf("number of core electrons           : %f\n", unit_cell().num_core_electrons());
    std::printf("number of valence electrons        : %f\n", unit_cell().num_valence_electrons());
    std::printf("total number of electrons          : %f\n", unit_cell().num_electrons());
    std::printf("extra charge                       : %f\n", parameters_input().extra_charge_);
    std::printf("total number of aw basis functions : %i\n", unit_cell().mt_aw_basis_size());
    std::printf("total number of lo basis functions : %i\n", unit_cell().mt_lo_basis_size());
    std::printf("number of first-variational states : %i\n", num_fv_states());
    std::printf("number of bands                    : %i\n", num_bands());
    std::printf("number of spins                    : %i\n", num_spins());
    std::printf("number of magnetic dimensions      : %i\n", num_mag_dims());
    std::printf("lmax_apw                           : %i\n", lmax_apw());
    std::printf("lmax_rho                           : %i\n", lmax_rho());
    std::printf("lmax_pot                           : %i\n", lmax_pot());
    std::printf("lmax_rf                            : %i\n", unit_cell_.lmax());
    std::printf("smearing width                     : %f\n", smearing_width());
    std::printf("cyclic block size                  : %i\n", cyclic_block_size());
    std::printf("|G+k| cutoff                       : %f\n", gk_cutoff());

    std::string reln[] = {"valence relativity                 : ", "core relativity                    : "};

    relativity_t relt[] = {valence_relativity_, core_relativity_};
    for (int i = 0; i < 2; i++) {
        std::printf("%s", reln[i].c_str());
        switch (relt[i]) {
            case relativity_t::none: {
                std::printf("none\n");
                break;
            }
            case relativity_t::koelling_harmon: {
                std::printf("Koelling-Harmon\n");
                break;
            }
            case relativity_t::zora: {
                std::printf("zora\n");
                break;
            }
            case relativity_t::iora: {
                std::printf("iora\n");
                break;
            }
            case relativity_t::dirac: {
                std::printf("Dirac\n");
                break;
            }
        }
    }

    std::string evsn[] = {"standard eigen-value solver        : ", "generalized eigen-value solver     : "};

    ev_solver_t evst[] = {std_evp_solver_type(), gen_evp_solver_type()};
    for (int i = 0; i < 2; i++) {
        std::printf("%s", evsn[i].c_str());
        switch (evst[i]) {
            case ev_solver_t::lapack: {
                std::printf("LAPACK\n");
                break;
            }
#if defined(__SCALAPACK)
            case ev_solver_t::scalapack: {
                std::printf("ScaLAPACK\n");
                break;
            }
#endif
#if defined(__ELPA)
            case ev_solver_t::elpa1: {
                std::printf("ELPA1\n");
                break;
            }
            case ev_solver_t::elpa2: {
                std::printf("ELPA2\n");
                break;
            }
#endif
#if defined(__MAGMA)
            case ev_solver_t::magma: {
                std::printf("MAGMA\n");
                break;
            }
            case ev_solver_t::magma_gpu: {
                std::printf("MAGMA with GPU pointers\n");
                break;
            }
#endif
            case ev_solver_t::plasma: {
                std::printf("PLASMA\n");
                break;
            }
#if defined(__CUDA)
            case ev_solver_t::cusolver: {
                std::printf("cuSOLVER\n");
                break;
            }
#endif
            default: {
                std::stringstream s;
                s << "wrong eigen-value solver: " << evsn[i];
                throw std::runtime_error(s.str());
            }
        }
    }

    std::printf("processing unit                    : ");
    switch (processing_unit()) {
        case device_t::CPU: {
            std::printf("CPU\n");
            break;
        }
        case device_t::GPU: {
            std::printf("GPU\n");
            acc::print_device_info(0);
            break;
        }
    }
    std::printf("\n");
    std::printf("iterative solver                   : %s\n", iterative_solver_input_.type_.c_str());
    std::printf("number of steps                    : %i\n", iterative_solver_input_.num_steps_);
    std::printf("subspace size                      : %i\n", iterative_solver_input_.subspace_size_);

    std::printf("\n");
    std::printf("spglib version: %d.%d.%d\n", spg_get_major_version(), spg_get_minor_version(), spg_get_micro_version());
    {
        unsigned int vmajor, vminor, vmicro;
        H5get_libversion(&vmajor, &vminor, &vmicro);
        std::printf("HDF5 version: %d.%d.%d\n", vmajor, vminor, vmicro);
    }
    {
        int vmajor, vminor, vmicro;
        xc_version(&vmajor, &vminor, &vmicro);
        std::printf("Libxc version: %d.%d.%d\n", vmajor, vminor, vmicro);
    }

    int i{1};
    std::printf("\n");
    std::printf("XC functionals\n");
    std::printf("==============\n");
    for (auto& xc_label : xc_functionals()) {
        XC_functional xc(spfft(), unit_cell().lattice_vectors(), xc_label, num_spins());
#if defined(__USE_VDWXC)
        if (xc.is_vdw()) {
            std::printf("Van der Walls functional\n");
            std::printf("%s\n", xc.refs().c_str());
            continue;
        }
#endif
        std::printf("%i) %s: %s\n", i, xc_label.c_str(), xc.name().c_str());
        std::printf("%s\n", xc.refs().c_str());
        i++;
    }

    if (!full_potential()) {
        std::printf("\n");
        std::printf("memory consumption\n");
        std::printf("==================\n");
        /* volume of the Brillouin zone */
        double v0 = std::pow(twopi, 3) / unit_cell().omega();
        /* volume of the cutoff sphere for wave-functions */
        double v1 = fourpi * std::pow(gk_cutoff(), 3) / 3;
        /* volume of the cutoff sphere for density and potential */
        double v2 = fourpi * std::pow(pw_cutoff(), 3) / 3;
        /* volume of the cutoff sphere for coarse FFT grid */
        double v3 = fourpi * std::pow(2 * gk_cutoff(), 3) / 3;
        /* approximate number of G+k vectors */
        auto ngk = static_cast<size_t>(v1 / v0);
        if (gamma_point()) {
            ngk /= 2;
        }
        /* approximate number of G vectors */
        auto ng = static_cast<size_t>(v2 / v0);
        if (control().reduce_gvec_) {
            ng /= 2;
        }
        /* approximate number of coarse G vectors */
        auto ngc = static_cast<size_t>(v3 / v0);
        if (control().reduce_gvec_) {
            ngc /= 2;
        }
        std::printf("approximate number of G+k vectors        : %li\n", ngk);
        std::printf("approximate number of G vectors          : %li\n", ng);
        std::printf("approximate number of coarse G vectors   : %li\n", ngc);
        size_t wf_size = ngk * num_bands() * num_spins() * 16;
        std::printf("approximate size of wave-functions for each k-point: %i Mb, %i Mb/rank\n",
            static_cast<int>(wf_size >> 20), static_cast<int>((wf_size / comm_band().size()) >> 20));

        /* number of simultaneously treated spin components */
        int num_sc = (num_mag_dims() == 3) ? 2 : 1;
        /* number of auxiliary basis functions */
        int num_phi = iterative_solver_input().subspace_size_ * num_bands();
        /* memory consumption for Davidson:
           - wave-functions psi (num_bands x num_spin)
           - Hpsi and Spsi (num_bands * num_sc)
           - auxiliary basis phi (num_bands x num_sc) and also Hphi and Sphi of the same size
           - residuals (num_bands * num_sc)
           - beta-projectors (estimated as num_bands)

           Each wave-function is of size ngk

           TODO: add estimation of subspace matrix size (H_{ij} and S_{ij})
        */
        size_t tot_size = (num_bands() * num_spins() + 2 * num_bands() * num_sc + 3 * num_phi * num_sc +
            num_bands() * num_sc + num_bands()) * ngk * sizeof(double_complex);
        std::printf("approximate memory consumption of Davidson solver: %i Mb/rank\n",
            static_cast<int>((tot_size / comm_band().size()) >> 20));

        if (unit_cell().augment()) {
            /* approximate size of local fraction of G vectors */
            size_t ngloc = std::max(static_cast<size_t>(1), ng / comm().size());
            /* upper limit of packed {xi,xi'} bete-projectors index */
            int nb = unit_cell().max_mt_basis_size() * (unit_cell().max_mt_basis_size() + 1) / 2;
            /* size of augmentation operator;
               factor 2 is needed for the estimation of GPU memory, as augmentation operator for two atom types
               will be stored on GPU and computation will be overlapped with transfer of the  next augmentation
               operator */
            // TODO: optimize generated_rho_aug() for less memory consumption
            size_t size_aug = nb * ngloc * sizeof(double_complex);
            if (unit_cell().num_atom_types() > 1) {
                size_aug *= 2;
            }

            /* and two more arrays will be allocated in generate_rho_aug() with 1Gb maximum size each */
            size_t size1 = nb * ngloc * sizeof(double_complex);
            size1 = std::min(size1, static_cast<size_t>(1 << 30));

            int max_atoms{0};
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                max_atoms = std::max(max_atoms, unit_cell().atom_type(iat).num_atoms());
            }
            size_t size2 = max_atoms * ngloc * sizeof(double_complex);
            size2 = std::min(size2, static_cast<size_t>(1 << 30));

            size_aug += (size1 + size2);
            std::printf("approximate memory consumption of charge density augmentation: %i Mb/rank\n",
                static_cast<int>(size_aug >> 20));
        }
        /* FFT buffers of fine and coarse meshes */
        size_t size_fft = spfft().local_slice_size() + spfft_coarse().local_slice_size();
        size_fft *= sizeof(double);
        if (!gamma_point()) {
            size_fft *= 2;
        }
        std::printf("approximate memory consumption of FFT transforms: %i Mb/rank\n",
            static_cast<int>(size_fft >> 20));
    }
}

/** The update of the lattice vectors or atomic positions has an impact on many quantities which have to be
    recomputed in the correct order. First, the unit cell is updated and the new reciprocal lattice vectors
    are obtained. Then the G-vectors are computed (if this is the first call to update()) or recomputed with a
    new reciprocal lattice, but without rebuilding a new list. On the first call the spfft objects are created
    after G-vectors. */
void Simulation_context::update()
{
    PROFILE("sirius::Simulation_context::update");

    /* update unit cell (reciprocal lattice, etc.) */
    unit_cell().update();

    /* create or update radial integrals */
    if (!full_potential()) {
        /* ratio of the unit cell volumes; if new volume is smaller than the initial, this ratio is > 1
           and we need to adjust the cutoff */
        double d = omega0_ / unit_cell().omega();
        double new_gk_cutoff = d * this->gk_cutoff();
        double new_pw_cutoff = d * this->pw_cutoff();

        /* radial integrals with pw_cutoff */
        if (!aug_ri_ || aug_ri_->qmax() < new_pw_cutoff) {
            aug_ri_ = std::unique_ptr<Radial_integrals_aug<false>>(
                new Radial_integrals_aug<false>(unit_cell(), new_pw_cutoff, settings().nprii_aug_));
        }

        if (!aug_ri_djl_ || aug_ri_djl_->qmax() < new_pw_cutoff) {
            aug_ri_djl_ = std::unique_ptr<Radial_integrals_aug<true>>(
                new Radial_integrals_aug<true>(unit_cell(), new_pw_cutoff, settings().nprii_aug_));
        }

        if (!ps_core_ri_ || ps_core_ri_->qmax() < new_pw_cutoff) {
            ps_core_ri_ = std::unique_ptr<Radial_integrals_rho_core_pseudo<false>>(
                new Radial_integrals_rho_core_pseudo<false>(unit_cell(), new_pw_cutoff, settings().nprii_rho_core_));
        }

        if (!ps_core_ri_djl_ || ps_core_ri_djl_->qmax() < new_pw_cutoff) {
            ps_core_ri_djl_ = std::unique_ptr<Radial_integrals_rho_core_pseudo<true>>(
                new Radial_integrals_rho_core_pseudo<true>(unit_cell(), new_pw_cutoff, settings().nprii_rho_core_));
        }

        if (!ps_rho_ri_ || ps_rho_ri_->qmax() < new_pw_cutoff) {
            ps_rho_ri_ = std::unique_ptr<Radial_integrals_rho_pseudo>(
                new Radial_integrals_rho_pseudo(unit_cell(), new_pw_cutoff, 20));
        }

        if (!vloc_ri_ || vloc_ri_->qmax() < new_pw_cutoff) {
            vloc_ri_ = std::unique_ptr<Radial_integrals_vloc<false>>(
                new Radial_integrals_vloc<false>(unit_cell(), new_pw_cutoff, settings().nprii_vloc_));
        }

        if (!vloc_ri_djl_ || vloc_ri_djl_->qmax() < new_pw_cutoff) {
            vloc_ri_djl_ = std::unique_ptr<Radial_integrals_vloc<true>>(
                new Radial_integrals_vloc<true>(unit_cell(), new_pw_cutoff, settings().nprii_vloc_));
        }

        /* radial integrals with pw_cutoff */
        if (!beta_ri_ || beta_ri_->qmax() < new_gk_cutoff) {
            beta_ri_ = std::unique_ptr<Radial_integrals_beta<false>>(
                new Radial_integrals_beta<false>(unit_cell(), new_gk_cutoff, settings().nprii_beta_));
        }

        if (!beta_ri_djl_ || beta_ri_djl_->qmax() < new_gk_cutoff) {
            beta_ri_djl_ = std::unique_ptr<Radial_integrals_beta<true>>(
                new Radial_integrals_beta<true>(unit_cell(), new_gk_cutoff, settings().nprii_beta_));
        }

        if (!atomic_wf_ri_ || atomic_wf_ri_->qmax() < new_gk_cutoff) {
            atomic_wf_ri_ = std::unique_ptr<Radial_integrals_atomic_wf<false>>(
                new Radial_integrals_atomic_wf<false>(unit_cell(), new_gk_cutoff, 20, false));
        }

        if (!atomic_wf_ri_djl_ || atomic_wf_ri_djl_->qmax() < new_gk_cutoff) {
            atomic_wf_ri_djl_ = std::unique_ptr<Radial_integrals_atomic_wf<true>>(
                new Radial_integrals_atomic_wf<true>(unit_cell(), new_gk_cutoff, 20, false));
        }

        if (!hubbard_wf_ri_ || hubbard_wf_ri_->qmax() < new_gk_cutoff) {
            hubbard_wf_ri_ = std::unique_ptr<Radial_integrals_atomic_wf<false>>(
                new Radial_integrals_atomic_wf<false>(unit_cell(), new_gk_cutoff, 20, true));
        }

        if (!hubbard_wf_ri_djl_ || hubbard_wf_ri_djl_->qmax() < new_gk_cutoff) {
            hubbard_wf_ri_djl_ = std::unique_ptr<Radial_integrals_atomic_wf<true>>(
                new Radial_integrals_atomic_wf<true>(unit_cell(), new_gk_cutoff, 20, true));
        }
    }

    /* get new reciprocal vector */
    auto rlv = unit_cell_.reciprocal_lattice_vectors();

    auto spfft_pu = this->processing_unit() == device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;

    /* create a list of G-vectors for corase FFT grid; this is done only once,
       the next time only reciprocal lattice of the G-vectors is updated */
    if (!gvec_coarse_) {
        /* create list of coarse G-vectors */
        gvec_coarse_ = std::unique_ptr<Gvec>(new Gvec(rlv, 2 * gk_cutoff(), comm(), control().reduce_gvec_));
        /* create FFT friendly partiton */
        gvec_coarse_partition_ = std::unique_ptr<Gvec_partition>(
            new Gvec_partition(*gvec_coarse_, comm_fft_coarse(), comm_ortho_fft_coarse()));

        auto spl_z = split_fft_z(fft_coarse_grid_[2], comm_fft_coarse());

        /* create spfft buffer for coarse transform */
        spfft_grid_coarse_ = std::unique_ptr<spfft::Grid>(
            new spfft::Grid(fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
                            gvec_coarse_partition_->zcol_count_fft(), spl_z.local_size(), spfft_pu, -1,
                            comm_fft_coarse().mpi_comm(), SPFFT_EXCH_DEFAULT));

        /* create spfft transformations */
        const auto fft_type_coarse = gvec_coarse().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto gv = gvec_coarse_partition_->get_gvec();

        /* create actual transform object */
        spfft_transform_coarse_.reset(new spfft::Transform(spfft_grid_coarse_->create_transform(
            spfft_pu, fft_type_coarse, fft_coarse_grid_[0], fft_coarse_grid_[1], fft_coarse_grid_[2],
            spl_z.local_size(), gvec_coarse_partition_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
            gv.at(memory_t::host))));
    } else {
        gvec_coarse_->lattice_vectors(rlv);
    }

    /* create a list of G-vectors for dense FFT grid; G-vectors are divided between all available MPI ranks.*/
    if (!gvec_) {
        gvec_ = std::unique_ptr<Gvec>(new Gvec(pw_cutoff(), *gvec_coarse_));
        gvec_partition_ = std::unique_ptr<Gvec_partition>(new Gvec_partition(*gvec_, comm_fft(), comm_ortho_fft()));

        auto spl_z = split_fft_z(fft_grid_[2], comm_fft());

        /* create spfft buffer for fine-grained transform */
        spfft_grid_ = std::unique_ptr<spfft::Grid>(
            new spfft::Grid(fft_grid_[0], fft_grid_[1], fft_grid_[2],
                            gvec_partition_->zcol_count_fft(), spl_z.local_size(), spfft_pu, -1,
                            comm_fft().mpi_comm(), SPFFT_EXCH_DEFAULT));

        const auto fft_type = gvec().reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

        auto gv = gvec_partition_->get_gvec();

        spfft_transform_.reset(new spfft::Transform(spfft_grid_->create_transform(
            spfft_pu, fft_type, fft_grid_[0], fft_grid_[1], fft_grid_[2],
            spl_z.local_size(), gvec_partition_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host))));

        /* copy G-vectors to GPU; this is done once because Miller indices of G-vectors
           do not change during the execution */
        switch (this->processing_unit()) {
            case device_t::CPU: {
                break;
            }
            case device_t::GPU: {
                gvec_coord_ = sddk::mdarray<int, 2>(gvec().count(), 3, memory_t::host, "gvec_coord_");
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec().count(); igloc++) {
                    int ig = gvec().offset() + igloc;
                    auto G = gvec().gvec(ig);
                    for (int x : {0, 1, 2}) {
                        gvec_coord_(igloc, x) = G[x];
                    }
                }
                gvec_coord_.allocate(memory_t::device).copy_to(memory_t::device);
                break;
            }
        }
    } else {
        gvec_->lattice_vectors(rlv);
    }

    if (!remap_gvec_) {
        remap_gvec_ = std::unique_ptr<Gvec_shells>(new Gvec_shells(gvec()));
    }

    /* check symmetry of G-vectors */
    if (unit_cell_.num_atoms() != 0 && use_symmetry() && control().verification_ >= 1) {
        check_gvec(gvec(), unit_cell_.symmetry());
        if (!full_potential()) {
            check_gvec(gvec_coarse(), unit_cell_.symmetry());
        }
        check_gvec(*remap_gvec_, unit_cell().symmetry());
    }

    /* check if FFT grid is OK; this check is especially needed if the grid is set as external parameter */
    if (control().verification_ >= 0) {
        #pragma omp parallel for
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            int ig = gvec().offset() + igloc;

            auto gv = gvec().gvec(ig);
            /* check limits */
            for (int x: {0, 1, 2}) {
                auto limits = fft_grid().limits(x);
                /* check boundaries */
                if (gv[x] < limits.first || gv[x] > limits.second) {
                    std::stringstream s;
                    s << "G-vector is outside of grid limits\n"
                      << "  G: " << gv << ", length: " << gvec().gvec_cart<index_domain_t::global>(ig).length() << "\n"
                      << "  FFT grid limits: "
                      << fft_grid().limits(0).first << " " << fft_grid().limits(0).second << " "
                      << fft_grid().limits(1).first << " " << fft_grid().limits(1).second << " "
                      << fft_grid().limits(2).first << " " << fft_grid().limits(2).second << "\n"
                      << "  FFT grid is not compatible with G-vector cutoff (" << this->pw_cutoff() << ")";
                    TERMINATE(s);
                }
            }
        }
    }

    init_atoms_to_grid_idx(control().rmt_max_);

    std::pair<int, int> limits(0, 0);
    for (int x : {0, 1, 2}) {
        limits.first  = std::min(limits.first, fft_grid().limits(x).first);
        limits.second = std::max(limits.second, fft_grid().limits(x).second);
    }

    /* recompute phase factors for atoms */
    phase_factors_ = mdarray<double_complex, 3>(3, limits, unit_cell().num_atoms(), memory_t::host, "phase_factors_");
    #pragma omp parallel for
    for (int i = limits.first; i <= limits.second; i++) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto pos = unit_cell_.atom(ia).position();
            for (int x : {0, 1, 2}) {
                phase_factors_(x, i, ia) = std::exp(double_complex(0.0, twopi * (i * pos[x])));
            }
        }
    }

    /* recompute phase factors for atom types */
    phase_factors_t_ = mdarray<double_complex, 2>(gvec().count(), unit_cell().num_atom_types());
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        /* global index of G-vector */
        int ig = gvec().offset() + igloc;
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            double_complex z(0, 0);
            for (int ia = 0; ia < unit_cell().atom_type(iat).num_atoms(); ia++) {
                z += gvec_phase_factor(ig, unit_cell().atom_type(iat).atom_id(ia));
            }
            phase_factors_t_(igloc, iat) = z;
        }
    }

    if (use_symmetry()) {
        sym_phase_factors_ = mdarray<double_complex, 3>(3, limits, unit_cell().symmetry().num_mag_sym());

        #pragma omp parallel for
        for (int i = limits.first; i <= limits.second; i++) {
            for (int isym = 0; isym < unit_cell().symmetry().num_mag_sym(); isym++) {
                auto t = unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.t;
                for (int x : {0, 1, 2}) {
                    sym_phase_factors_(x, i, isym) = std::exp(double_complex(0.0, twopi * (i * t[x])));
                }
            }
        }
    }

    /* precompute some G-vector related arrays */
    gvec_tp_ = sddk::mdarray<double, 2>(gvec().count(), 2, memory_t::host, "gvec_tp_");
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < gvec().count(); igloc++) {
        auto rtp = SHT::spherical_coordinates(gvec().gvec_cart<index_domain_t::local>(igloc));
        gvec_tp_(igloc, 0) = rtp[1];
        gvec_tp_(igloc, 1) = rtp[2];
    }

    switch (this->processing_unit()) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
            gvec_tp_.allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
    }

    if (full_potential()) {
        init_step_function();
    }

    if (!full_potential()) {
        memory_pool* mp{nullptr};
        memory_pool* mpd{nullptr};
        switch (this->processing_unit()) {
            case device_t::CPU: {
                mp = &mem_pool(memory_t::host);
                break;
            }
            case device_t::GPU: {
                mp  = &mem_pool(memory_t::host_pinned);
                mpd = &mem_pool(memory_t::device);
                break;
            }
        }
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            if (unit_cell().atom_type(iat).augment() && unit_cell().atom_type(iat).num_atoms() > 0) {
                augmentation_op_[iat] = std::unique_ptr<Augmentation_operator>(
                    new Augmentation_operator(unit_cell().atom_type(iat), gvec()));
                augmentation_op_[iat]->generate_pw_coeffs(aug_ri(), gvec_tp_, *mp, mpd);

            } else {
                augmentation_op_[iat] = nullptr;
            }
        }
    }
}

void Simulation_context::create_storage_file() const
{
    if (comm_.rank() == 0) {
        /* create new hdf5 file */
        HDF5_tree fout(storage_file_name, hdf5_access_t::truncate);
        fout.create_node("parameters");
        fout.create_node("effective_potential");
        fout.create_node("effective_magnetic_field");
        fout.create_node("density");
        fout.create_node("magnetization");

        for (int j = 0; j < num_mag_dims(); j++) {
            fout["magnetization"].create_node(j);
            fout["effective_magnetic_field"].create_node(j);
        }

        fout["parameters"].write("num_spins", num_spins());
        fout["parameters"].write("num_mag_dims", num_mag_dims());
        fout["parameters"].write("num_bands", num_bands());

        mdarray<int, 2> gv(3, gvec().num_gvec());
        for (int ig = 0; ig < gvec().num_gvec(); ig++) {
            auto G = gvec().gvec(ig);
            for (int x : {0, 1, 2}) {
                gv(x, ig) = G[x];
            }
        }
        fout["parameters"].write("num_gvec", gvec().num_gvec());
        fout["parameters"].write("gvec", gv);

        fout.create_node("unit_cell");
        fout["unit_cell"].create_node("atoms");
        for (int j = 0; j < unit_cell().num_atoms(); j++) {
            fout["unit_cell"]["atoms"].create_node(j);
            fout["unit_cell"]["atoms"][j].write("mt_basis_size", unit_cell().atom(j).mt_basis_size());
        }
    }
    comm_.barrier();
}

void Simulation_context::generate_phase_factors(int iat__, mdarray<double_complex, 2> &phase_factors__) const 
{
    PROFILE("sirius::Simulation_context::generate_phase_factors");
    int na = unit_cell_.atom_type(iat__).num_atoms();
    switch (processing_unit_) {
        case device_t::CPU: {
            #pragma omp parallel for
            for (int igloc = 0; igloc < gvec().count(); igloc++) {
                int ig = gvec().offset() + igloc;
                for (int i = 0; i < na; i++) {
                    int ia                    = unit_cell().atom_type(iat__).atom_id(i);
                    phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                }
            }
            break;
        }
        case device_t::GPU: {
#if defined(__GPU)
            generate_phase_factors_gpu(gvec().count(), na, gvec_coord().at(memory_t::device),
                                       unit_cell().atom_coord(iat__).at(memory_t::device), phase_factors__.at(memory_t::device));
#endif
            break;
        }
    }
}

void Simulation_context::print_memory_usage(const char *file__, int line__)
{
    if (comm().rank() == 0 && control().print_memory_usage_ && control().verbosity_ >= 1) {
        sirius::print_memory_usage(file__, line__);

        std::vector<std::string> labels = {"host"};
        std::vector<memory_pool*> mp = {&this->mem_pool(memory_t::host)};
        int np{1};
        if (processing_unit() == device_t::GPU) {
            labels.push_back("host pinned");
            labels.push_back("device");
            mp.push_back(&this->mem_pool(memory_t::host_pinned));
            mp.push_back(&this->mem_pool(memory_t::device));
            np = 3;
        }
        std::printf("memory pools\n");
        std::printf("------------\n");
        for (int i = 0; i < np; i++) {
            std::printf("%s: total capacity: %li Mb, free: %li Mb, num.blocks: %li, num.pointers: %li\n",
                labels[i].c_str(), mp[i]->total_size() >> 20, mp[i]->free_size() >> 20, mp[i]->num_blocks(),
                mp[i]->num_stored_ptr());
        }
    }
}

void Simulation_context::init_atoms_to_grid_idx(double R__)
{
    PROFILE("sirius::Simulation_context::init_atoms_to_grid_idx");

    atoms_to_grid_idx_.resize(unit_cell_.num_atoms());

    vector3d<double> delta(1.0 / spfft().dim_x(), 1.0 / spfft().dim_y(), 1.0 / spfft().dim_z());

    int z_off = spfft().local_z_offset();
    vector3d<int> grid_beg(0, 0, z_off);
    vector3d<int> grid_end(spfft().dim_x(), spfft().dim_y(), z_off + spfft().local_z_length());
    std::vector<vector3d<double>> verts_cart{{-R__, -R__, -R__}, {R__, -R__, -R__}, {-R__, R__, -R__},
                                             {R__, R__, -R__},   {-R__, -R__, R__}, {R__, -R__, R__},
                                             {-R__, R__, R__},   {R__, R__, R__}};

    auto bounds_box = [&](vector3d<double> pos) {
        std::vector<vector3d<double>> verts;

        /* pos is a position of atom */
        for (auto v : verts_cart) {
            verts.push_back(pos + unit_cell_.get_fractional_coordinates(v));
        }

        std::pair<vector3d<int>, vector3d<int>> bounds_ind;

        for (int x : {0, 1, 2}) {
            std::sort(verts.begin(), verts.end(),
                      [x](vector3d<double>& a, vector3d<double>& b) { return a[x] < b[x]; });
            bounds_ind.first[x]  = std::max(static_cast<int>(verts[0][x] / delta[x]) - 1, grid_beg[x]);
            bounds_ind.second[x] = std::min(static_cast<int>(verts[5][x] / delta[x]) + 1, grid_end[x]);
        }

        return bounds_ind;
    };

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {

        std::vector<std::pair<int, double>> atom_to_ind_map;

        for (int t0 = -1; t0 <= 1; t0++) {
            for (int t1 = -1; t1 <= 1; t1++) {
                for (int t2 = -1; t2 <= 1; t2++) {
                    auto pos = unit_cell_.atom(ia).position() + vector3d<double>(t0, t1, t2);

                    /* find the small box around this atom */
                    auto box = bounds_box(pos);

                    for (int j0 = box.first[0]; j0 < box.second[0]; j0++) {
                        for (int j1 = box.first[1]; j1 < box.second[1]; j1++) {
                            for (int j2 = box.first[2]; j2 < box.second[2]; j2++) {
                                auto v = pos - vector3d<double>(delta[0] * j0, delta[1] * j1, delta[2] * j2);
                                auto r  = unit_cell_.get_cartesian_coordinates(v).length();
                                if (r < R__) {
                                    auto ir = fft_grid_.index_by_coord(j0, j1, j2 - z_off);
                                    atom_to_ind_map.push_back({ir, r});
                                }
                            }
                        }
                    }
                }
            }
        }

        atoms_to_grid_idx_[ia] = std::move(atom_to_ind_map);
    }
}

void Simulation_context::init_step_function()
{
    auto v = make_periodic_function<index_domain_t::global>([&](int iat, double g)
                                                            {
                                                                auto R = unit_cell().atom_type(iat).mt_radius();
                                                                return unit_step_function_form_factors(R, g);
                                                            });

    theta_    = mdarray<double, 1>(spfft().local_slice_size());
    theta_pw_ = mdarray<double_complex, 1>(gvec().num_gvec());

    for (int ig = 0; ig < gvec().num_gvec(); ig++) {
        theta_pw_[ig] = -v[ig];
    }
    theta_pw_[0] += 1.0;

    std::vector<double_complex> ftmp(gvec_partition().gvec_count_fft());
    for (int i = 0; i < gvec_partition().gvec_count_fft(); i++) {
        ftmp[i] = theta_pw_[gvec_partition().idx_gvec(i)];
    }
    spfft().backward(reinterpret_cast<double const*>(ftmp.data()), SPFFT_PU_HOST);
    spfft_output(spfft(), &theta_[0]);

    double vit{0};
    for (int i = 0; i < spfft().local_slice_size(); i++) {
        vit += theta_[i];
    }
    vit *= (unit_cell().omega() / fft_grid().num_points());
    Communicator(spfft().communicator()).allreduce(&vit, 1);

    if (std::abs(vit - unit_cell().volume_it()) > 1e-10) {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << std::abs(vit - unit_cell().volume_it());
        if (comm().rank() == 0) {
            WARNING(s);
        }
    }
    if (control().print_checksum_) {
        double_complex z1 = theta_pw_.checksum();
        double d1         = theta_.checksum();
        Communicator(spfft().communicator()).allreduce(&d1, 1);
        if (comm().rank() == 0) {
            utils::print_checksum("theta", d1);
            utils::print_checksum("theta_pw", z1);
        }
    }
}

void Simulation_context::init_comm()
{
    PROFILE("sirius::Simulation_context::init_comm");

    /* check MPI grid dimensions and set a default grid if needed */
    if (!control().mpi_grid_dims_.size()) {
        mpi_grid_dims({1, 1});
    }
    if (control().mpi_grid_dims_.size() != 2) {
        TERMINATE("wrong MPI grid");
    }

    int npr = control_input_.mpi_grid_dims_[0];
    int npc = control_input_.mpi_grid_dims_[1];
    int npb = npr * npc;
    int npk = comm_.size() / npb;
    if (npk * npb != comm_.size()) {
        std::stringstream s;
        s << "Can't divide " << comm_.size() << " ranks into groups of size " << npb;
        TERMINATE(s);
    }

    /* setup MPI grid */
    mpi_grid_ = std::unique_ptr<MPI_grid>(new MPI_grid({npk, npc, npr}, comm_));

    /* here we know the number of ranks for band parallelization */

    /* if we have multiple ranks per node and band parallelization, switch to parallel FFT for coarse mesh */
    if (num_ranks_per_node() > 1 && comm_band().size() > 1) {
        control_input_.fft_mode_ = "parallel";
    }

    /* create communicator, orthogonal to comm_fft_coarse */
    comm_ortho_fft_coarse_ = comm().split(comm_fft_coarse().rank());

    /* create communicator, orthogonal to comm_fft_coarse within a band communicator */
    comm_band_ortho_fft_coarse_ = comm_band().split(comm_fft_coarse().rank());
}

} // namespace sirius

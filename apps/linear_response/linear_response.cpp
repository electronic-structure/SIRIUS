#include "utils/profiler.hpp"
#include <sirius.hpp>
#include "../dft_loop/filesystem.hpp"
#include <utils/json.hpp>
#include <cfenv>
#include <fenv.h>

#include "band/residuals.hpp"
#include "SDDK/wf_trans.hpp"

#include "multi_cg/multi_cg.hpp"

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new     = 0,
    ground_state_restart = 1,
};

void rewrite_relative_paths(json& dict__, fs::path const &working_directory = fs::current_path())
{
    // the json.unit_cell.atom_files[] dict might contain relative paths,
    // which should be relative to the json file. So better make them
    // absolute such that the simulation context does not have to be
    // aware of paths.
    if (!dict__.count("unit_cell"))
        return;

    auto &section = dict__["unit_cell"];

    if (!section.count("atom_files"))
        return;

    auto &atom_files = section["atom_files"];

    for (auto& label : atom_files.items()) {
        label.value() = working_directory / std::string(label.value());
    }
}

nlohmann::json preprocess_json_input(std::string fname__)
{
    if (fname__.find("{") == std::string::npos) {
        // If it's a file, set the working directory to that file.
        auto json = utils::read_json_from_file(fname__);
        rewrite_relative_paths(json, fs::path{fname__}.parent_path());
        return json;
    } else {
        // Raw JSON input
        auto json = utils::read_json_from_string(fname__);
        rewrite_relative_paths(json);
        return json;
    }
}

std::unique_ptr<Simulation_context> create_sim_ctx(std::string fname__,
                                                   cmd_args const& args__)
{

    auto json = preprocess_json_input(fname__);

    auto ctx_ptr = std::make_unique<Simulation_context>(json.dump(), Communicator::world());
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.cfg().parameters();
    if (inp.gamma_point() && !(inp.ngridk()[0] * inp.ngridk()[1] * inp.ngridk()[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    ctx.import(args__);

    return ctx_ptr;
}

struct Wave_functions_wrap {
    Wave_functions<double> *x;

    typedef double_complex value_type;

    void fill(double_complex val) {
        x->pw_coeffs(0).prime() = [=](){
            return val;
        };
    }

    int cols() const {
        return x->num_wf();
    }

    void block_dot(Wave_functions_wrap const &y, std::vector<double_complex> &rhos, size_t num_unconverged) {
        auto result = x->dot(device_t::CPU, sddk::spin_range(0), *y.x, static_cast<int>(num_unconverged));
        for (int i = 0; i < static_cast<int>(num_unconverged); ++i)
            rhos[i] = result(i);
    }

    void repack(std::vector<size_t> const &ids) {
        size_t j = 0;
        for (auto i : ids) {
            if (j != i) {
                x->copy_from(*x, 1, 0, i, 0, j);
            }

            ++j;
        }
    }

    void copy(Wave_functions_wrap const &y, size_t num) {
        x->copy_from(*y.x, static_cast<int>(num), 0, 0, 0, 0);
    }

    void block_xpby(Wave_functions_wrap const &y, std::vector<double_complex> const &alphas, size_t num) {
        x->xpby(device_t::CPU, sddk::spin_range(0), *y.x, alphas, static_cast<int>(num));
    }

    void block_axpy_scatter(std::vector<double_complex> const &alphas, Wave_functions_wrap const &y, std::vector<size_t> const &ids) {
        x->axpy_scatter(device_t::CPU, sddk::spin_range(0), alphas, *y.x, ids);
    }

    void block_axpy(std::vector<double_complex> const &alphas, Wave_functions_wrap const &y, size_t num) {
        x->axpy(device_t::CPU, sddk::spin_range(0), alphas, *y.x, static_cast<int>(num));
    }
};

struct identity_preconditioner {
    size_t num_active;

    void apply(Wave_functions_wrap &x, Wave_functions_wrap const &y) {
        x.copy(y, num_active);
    }

    void repack(std::vector<size_t> const &ids) {
        num_active = ids.size();
    }
};

struct projector_H_min_e_S_projector {
    Hamiltonian_k<double> Hk;
    std::vector<double> min_eigenvals;
    Wave_functions<double> * Hphi;
    Wave_functions<double> * Sphi;

    projector_H_min_e_S_projector(Hamiltonian_k<double> && Hk, std::vector<double> const &eigvals, Wave_functions<double> * Hphi, Wave_functions<double> * Sphi)
    : Hk(std::move(Hk)), min_eigenvals(eigvals), Hphi(Hphi), Sphi(Sphi)
    {
        // flip the sign of the eigenvals so that the axpby works
        for (auto &e : min_eigenvals)
            e *= -1;
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            min_eigenvals[i] = min_eigenvals[ids[i]];
        }
    }

    // y[:, i] <- alpha * A * x[:, i] + beta * y[:, i] where A = (H - e_j S)
    void multiply(double alpha, Wave_functions_wrap x, double beta, Wave_functions_wrap y, int num_active) {
        // Hphi = H * x, Sphi = S * x
        Hk.apply_h_s<double_complex>(
            spin_range(0),
            0,
            num_active,
            *x.x,
            Hphi,
            Sphi
        );

        // Sphi[:, i] = Hphi[:, i] + min_eigenvals[:, i] * Sphi[:, i]
        Sphi->xpby(device_t::CPU, spin_range(0), *Hphi, min_eigenvals, num_active);

        // y[:, i] <- alpha * Sphi[:, i] + beta * y[:, i]
        y.x->axpby(device_t::CPU, spin_range(0), alpha, *Sphi, beta, num_active);
    }
};

void ground_state(Simulation_context& ctx,
                    task_t              task,
                    cmd_args const&     args,
                    int                 write_output)
{
    auto& inp = ctx.cfg().parameters();

    if (ctx.num_mag_dims() != 1)
        return;

    bool const reduce_kp = ctx.use_symmetry() && ctx.cfg().parameters().use_ibz();
    K_point_set kset(ctx, ctx.cfg().parameters().ngridk(), ctx.cfg().parameters().shiftk(), reduce_kp);
    DFT_ground_state dft(kset);

    auto& potential = dft.potential();
    dft.initial_state();

    /* launch the calculation */
    auto result = dft.find(inp.density_tol(), inp.energy_tol(), ctx.cfg().iterative_solver().energy_tolerance(),
            inp.num_dft_iter(), false);

    // now do something linear responsy.
    Hamiltonian0<double> H0(potential);

    for (int ikloc = 0; ikloc < kset.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset.spl_num_kpoints(ikloc);
        auto kp = kset.get<double>(ik);
        auto &Q = kp->spinor_wave_functions();
        auto num_sc = Q.num_sc();
        auto num_wf = Q.num_wf();
        auto& mp = ctx.mem_pool(ctx.host_memory_t());

        Wave_functions<double> Hphi(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);
        Wave_functions<double> Sphi(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);

        auto H_min_e_times_S = projector_H_min_e_S_projector(
            H0(*kp),
            kset.get_band_energies<double>(ik, 0),
            &Hphi,
            &Sphi
        );

        // Wave_functions Hphi(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);
        // Wave_functions res(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);

        // H_min_e_times_S.Hk.apply_h_s<double_complex>(
        //     spin_range(0), 0, num_wf, Q, &Hphi, &Sphi
        // );

        // mdarray<double, 1> eval(num_wf);
        // for (int i = 0; i < num_wf; ++i)
        //     eval(i) = H_min_e_times_S.eigenvals[i];

        // sirius::compute_residuals(
        //     ctx.aux_preferred_memory_t(),
        //     spin_range(0),
        //     num_wf,
        //     eval,
        //     Hphi,
        //     Sphi,
        //     res
        // );

        // auto result = res.l2norm(device_t::CPU, spin_range(0), num_wf);

        // for (int i = 0; i < num_wf; ++i) {
        //     std::cout << "i = " << i << ": " << result[i] << '\n';
        // }

        identity_preconditioner preconditioner{static_cast<size_t>(num_wf)};

        // State vectors for CG, where the right hand side B gets overwritten in place
        // X is the solution, and U and C are auxiliary.
        Wave_functions<double> X(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);
        Wave_functions<double> B(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);
        Wave_functions<double> U(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);
        Wave_functions<double> C(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);

        // When applying projectors, we need S * Q to be available.
        Wave_functions<double> SQ(mp, kp->gkvec_partition(), num_wf, ctx.aux_preferred_memory_t(), num_sc);

        sirius::apply_S_operator<double_complex>(
            ctx.processing_unit(),
            spin_range(0),
            0,
            num_wf,
            kp->beta_projectors(),
            Q,
            &H0.Q(),
            SQ);

        // Initial guess should be orthogonal to psi, so 0 works.
        X.zero(device_t::CPU, 0, 0, num_wf);

        // Create a constant right-hand side.
        B.pw_coeffs(0).prime() = []() { return 1.0; };

        // Apply B <- (I - SQQ')B
        sddk::dmatrix<double_complex> ovlp(num_wf, num_wf);
        sddk::inner(ctx.spla_context(), spin_range(0), Q, 0, num_wf, B, 0, num_wf, ovlp, 0, 0);
        sddk::transform<double>(ctx.spla_context(), 0, -1.0, {&SQ}, 0, num_wf, ovlp, 0, 0, 1.0, {&B}, 0, num_wf);

        // check if things are block-orthogonal.
        sddk::inner(ctx.spla_context(), spin_range(0), Q, 0, num_wf, B, 0, num_wf, ovlp, 0, 0);

        auto X_wrap = Wave_functions_wrap{&X};
        auto B_wrap = Wave_functions_wrap{&B};
        auto U_wrap = Wave_functions_wrap{&U};
        auto C_wrap = Wave_functions_wrap{&C};

        sirius::cg::multi_cg(
            H_min_e_times_S,
            preconditioner,
            X_wrap,
            B_wrap,
            U_wrap,
            C_wrap
        );
    }

    /* wait for all */
    ctx.comm().barrier();
}

/// Run a task based on a command line input.
void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));

    /* get the input file name */
    auto fpath = args.value<fs::path>("input", "sirius.json");

    if (fs::is_directory(fpath)) {
        fpath /= "sirius.json";
    }

    if (!fs::exists(fpath)) {
        if (Communicator::world().rank() == 0) {
            std::printf("input file does not exist\n");
        }
        return;
    }

    auto fname = fpath.string();

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        ground_state(*ctx, task, args, 1);
    }
}

int main(int argn, char** argv)
{
    std::feclearexcept(FE_ALL_EXCEPT);
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--output=", "{string} output file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");
    args.register_key("--repeat_update=", "{int} number of times to repeat update()");
    args.register_key("--fpe", "enable check of floating-point exceptions using GNUC library");
    args.register_key("--control.processing_unit=", "");
    args.register_key("--control.verbosity=", "");
    args.register_key("--control.verification=", "");
    args.register_key("--control.mpi_grid_dims=","");
    args.register_key("--control.std_evp_solver_name=", "");
    args.register_key("--control.gen_evp_solver_name=", "");
    args.register_key("--control.fft_mode=", "");
    args.register_key("--control.memory_usage=", "");
    args.register_key("--parameters.ngridk=", "");
    args.register_key("--parameters.gamma_point=", "");
    args.register_key("--parameters.pw_cutoff=", "");
    args.register_key("--iterative_solver.orthogonalize=", "");
    args.register_key("--iterative_solver.early_restart=", "{double} value between 0 and 1 to control the early restart ratio in Davidson");
    args.register_key("--mixer.type=", "{string} mixer name (anderson, anderson_stable, broyden2, linear)");
    args.register_key("--mixer.beta=", "{double} mixing parameter");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

#if defined(_GNU_SOURCE)
    if (args.exist("fpe")) {
        feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
    }
#endif

    sirius::initialize(1);

    run_tasks(args);

    int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                          rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median, rt_graph::Stat::Min,
                                          rt_graph::Stat::Max});
        std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        ofs << timing_result.json();
    }
    if (std::fetestexcept(FE_DIVBYZERO)) {
        std::cout << "FE_DIVBYZERO exception\n";
    }
    if (std::fetestexcept(FE_INVALID)) {
        std::cout << "FE_INVALID exception\n";
    }
    if (std::fetestexcept(FE_UNDERFLOW)) {
        std::cout << "FE_UNDERFLOW exception\n";
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        std::cout << "FE_OVERFLOW exception\n";
    }

    return 0;
}

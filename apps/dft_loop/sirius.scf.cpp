#include <sirius.h>
#include <json.hpp>

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

class lattice3d
{
    private:

        /// Bravais lattice vectors in column order.
        /** The following convention is used to transform fractional coordinates to Cartesian:  
         *  \f[
         *    \vec v_{C} = {\bf L} \vec v_{f}
         *  \f]
         */
        matrix3d<double> lattice_vectors_;

    public:
        
        lattice3d(vector3d<double> a0__,
                  vector3d<double> a1__,
                  vector3d<double> a2__)
        {
            for (int x: {0, 1, 2}) {
                lattice_vectors_(x, 0) = a0__[x];
                lattice_vectors_(x, 1) = a1__[x];
                lattice_vectors_(x, 2) = a2__[x];
            }
        }

        lattice3d(double a__,
                  double b__,
                  double c__,
                  double alpha__,
                  double beta__,
                  double gamma__)
        {
            lattice_vectors_(0, 0) = a__;
            lattice_vectors_(1, 0) = 0.0;
            lattice_vectors_(2, 0) = 0.0;

            lattice_vectors_(0, 1) = b__ * std::cos(gamma__);
            lattice_vectors_(1, 1) = b__ * std::sin(gamma__);
            lattice_vectors_(2, 1) = 0.0;

            lattice_vectors_(0, 2) = c__ * std::cos(beta__);
            lattice_vectors_(1, 2) = c__ * (std::cos(alpha__) - std::cos(gamma__) * std::cos(beta__)) / std::sin(gamma__);
            lattice_vectors_(2, 2) = std::sqrt(c__ * c__ - std::pow(lattice_vectors_(0, 2), 2) - std::pow(lattice_vectors_(1, 2), 2));
        }

        unit_cell_parameters_descriptor parameters()
        {
            unit_cell_parameters_descriptor d;
        
            vector3d<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
            vector3d<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
            vector3d<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));
        
            d.a = v0.length();
            d.b = v1.length();
            d.c = v2.length();
        
            d.alpha = std::acos((v1 * v2) / d.b / d.c);
            d.beta  = std::acos((v0 * v2) / d.a / d.c);
            d.gamma = std::acos((v0 * v1) / d.a / d.b);
        
            return d;
        }
};

class Parameter_optimization
{
    private:

        double step_;

        bool can_increase_step_{true};

        std::vector<double> x_;

        std::vector<double> f_;

        std::vector<double> df_;

        std::vector<int> df_sign_change_;

    public:

        Parameter_optimization(double step__)
            : step_(step__)
        {
        }
        
        void add_point(double x__, double f__, double df__)
        {
            if (df_.size() > 0) {
                if (df_.back() * df__ > 0) {
                    if (can_increase_step_) {
                        step_ *= 1.5;
                    }
                } else {
                    step_ *= 0.25;
                    can_increase_step_ = false;
                    df_sign_change_.push_back(static_cast<int>(df_.size()));
                }
            }
            
            x_.push_back(x__);
            f_.push_back(f__);
            df_.push_back(df__);
        }

        double next_x()
        {
            return x_.back() - Utils::sign(df_.back()) * step_;
        }

        void estimate_x0()
        {
            //for (int i = *(df_sign_change_.end() - 4); i < df_.size(); i++) {
            for (int i = df_sign_change_[0]; i < (int)df_.size(); i++) {
                printf("%18.10f %18.10f\n", x_[i], df_[i]);
            }
        }

        double step() const
        {
            return step_;
        }

};

enum class task_t
{
    ground_state_new = 0,
    ground_state_restart = 1,
    relaxation_new = 2,
    relaxation_restart = 3,
    lattice_relaxation_new = 4,
    volume_relaxation_new = 5,
    volume_relaxation_descent = 6
};

const double au2angs = 0.5291772108;

void json_output_common(json& dict__)
{
    dict__["git_hash"] = git_hash;
    dict__["build_date"] = build_date;
    dict__["comm_world_size"] = mpi_comm_world().size();
    dict__["threads_per_rank"] = omp_get_max_threads();
}

std::unique_ptr<Simulation_context> create_sim_ctx(std::string                     fname__,
                                                   cmd_args const&                 args__,
                                                   Parameters_input_section const& inp__)
{
    Simulation_context* ctx_ptr = new Simulation_context(fname__, mpi_comm_world());
    Simulation_context& ctx = *ctx_ptr;

    std::vector<int> mpi_grid_dims = ctx.mpi_grid_dims();
    mpi_grid_dims = args__.value< std::vector<int> >("mpi_grid", mpi_grid_dims);
    ctx.set_mpi_grid_dims(mpi_grid_dims);

    ctx.set_esm_type(inp__.esm_);
    ctx.set_num_fv_states(inp__.num_fv_states_);
    ctx.set_smearing_width(inp__.smearing_width_);
    for (auto& s: inp__.xc_functionals_) {
        ctx.add_xc_functional(s);
    }
    ctx.set_pw_cutoff(inp__.pw_cutoff_);
    ctx.set_aw_cutoff(inp__.aw_cutoff_);
    ctx.set_gk_cutoff(inp__.gk_cutoff_);
    if (ctx.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        ctx.set_lmax_apw(inp__.lmax_apw_);
        ctx.set_lmax_pot(inp__.lmax_pot_);
        ctx.set_lmax_rho(inp__.lmax_rho_);
    }
    ctx.set_num_mag_dims(inp__.num_mag_dims_);
    ctx.set_auto_rmt(inp__.auto_rmt_);
    ctx.set_core_relativity(inp__.core_relativity_);
    ctx.set_valence_relativity(inp__.valence_relativity_);
    ctx.set_gamma_point(inp__.gamma_point_);
    ctx.set_molecule(inp__.molecule_);

    return std::move(std::unique_ptr<Simulation_context>(ctx_ptr));
}

double ground_state(Simulation_context&       ctx,
                    task_t                    task,
                    cmd_args const&           args,
                    Parameters_input_section& inp,
                    int                       write_output)
{
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif
    
    Potential potential(ctx);
    potential.allocate();

    Density density(ctx);
    density.allocate();

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    K_set ks(ctx, ctx.mpi_grid().communicator(1 << _mpi_dim_k_), inp.ngridk_, inp.shiftk_, inp.use_symmetry_);
    ks.initialize();
    
    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    std::string ref_file = args.value<std::string>("test_against", "");
    bool write_state = (ref_file.size() == 0);
    
    DFT_ground_state dft(ctx, potential, density, ks, inp.use_symmetry_);

    if (task == task_t::ground_state_restart) {
        if (!Utils::file_exists(storage_file_name)) {
            TERMINATE("storage file is not found");
        }
        density.load();
        potential.load();
    } else {
        density.initial_density();
        dft.generate_effective_potential();
        if (!ctx.full_potential()) {
            dft.initialize_subspace();
        }
    }
    
    int result = dft.find(inp.potential_tol_, inp.energy_tol_, inp.num_dft_iter_, write_state);

    if (ref_file.size() != 0) {
        json dict;
        dict["ground_state"] = dft.serialize();
        json dict_ref;
        std::ifstream(ref_file) >> dict_ref;
        
        double e1 = dict["ground_state"]["energy"]["total"];
        double e2 = dict_ref["ground_state"]["energy"]["total"];

        if (std::abs(e1 - e2) > 1e-8) {
            printf("total energy is different\n");
            exit(1);
        }

        write_output = 0;
    }
    
    if (!ctx.full_potential()) {
        dft.forces();
    }

    if (write_output) {
        json dict;
        json_output_common(dict);
        
        dict["task"] = static_cast<int>(task);
        dict["ground_state"] = dft.serialize();
        dict["timers"] = runtime::Timer::serialize();
 
        if (ctx.comm().rank() == 0) {
            std::ofstream ofs(std::string("output_") + ctx.start_time_tag() + std::string(".json"),
                              std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }
        
        if (args.exist("aiida_output")) {
            json dict;
            json_output_common(dict);
            dict["task"] = static_cast<int>(task);
            if (result >= 0) {
                dict["task_status"] = "converged";
                dict["num_scf_iterations"] =  result;
            } else {
                dict["task_status"] = "unconverged";
            }
            dict["volume"] = ctx.unit_cell().omega() * std::pow(au2angs, 3);
            dict["volume_units"] = "angstrom^3";
            dict["energy"] = dft.total_energy() * ha2ev;
            dict["energy_units"] = "eV";
            if (ctx.comm().rank() == 0) {
                std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
                ofs << dict.dump(4);
            }
        }
    }

    /* wait for all */
    ctx.comm().barrier();

    runtime::Timer::print();

    return dft.total_energy();
}

double Etot_fake(vector3d<double> a0__, vector3d<double> a1__, vector3d<double> a2__)
{
    vector3d<double> a0{0, 5, 5};
    vector3d<double> a1{5, 0, 5};
    vector3d<double> a2{5, 5, 0};

    return std::pow((a0 - a0__).length(), 2) + std::pow((a1 - a1__).length(), 2) + std::pow((a2 - a2__).length(), 2);
}

double Etot_fake(double a__, double b__, double c__, double alpha__, double beta__, double gamma__)
{
    vector3d<double> a0{0, 5, 5};
    vector3d<double> a1{5, 0, 5};
    vector3d<double> a2{5, 5, 0};

    lattice3d ref_lat(a0, a1, a2);

    auto ref_lat_param = ref_lat.parameters();

    return std::pow(a__ - ref_lat_param.a, 2) +
           std::pow(b__ - ref_lat_param.b, 2) +
           std::pow(c__ - ref_lat_param.c, 2) + 
           std::pow(std::abs(alpha__ - ref_lat_param.alpha), 2) +
           std::pow(std::abs(beta__ - ref_lat_param.beta), 2) +
           std::pow(std::abs(gamma__ - ref_lat_param.gamma), 2);
}

void lattice_relaxation(task_t task, cmd_args args, Parameters_input_section& inp)
{
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");

    std::unique_ptr<Simulation_context> ctx0(create_sim_ctx(fname, args, inp));

    auto a0 = ctx0->unit_cell().lattice_vector(0);
    auto a1 = ctx0->unit_cell().lattice_vector(1);
    auto a2 = ctx0->unit_cell().lattice_vector(2);

    auto lat_param = lattice3d(a0, a1, a2).parameters();

    std::vector<double> params({lat_param.a,
                                lat_param.b,
                                lat_param.c,
                                lat_param.alpha,
                                lat_param.beta,
                                lat_param.gamma});

    std::vector<Parameter_optimization> param_opt({Parameter_optimization(0.25),
                                                   Parameter_optimization(0.25),
                                                   Parameter_optimization(0.25),
                                                   Parameter_optimization(0.1),
                                                   Parameter_optimization(0.1),
                                                   Parameter_optimization(0.1)});


    for (int iter = 0; iter < 100; iter++) {
        std::vector<double> p0 = params;
        double e0 = Etot_fake(p0[0], p0[1], p0[2], p0[3], p0[4], p0[5]);

        for (int i = 0; i < 6; i++) {
            double step = 1e-5;
            std::vector<double> p1 = params;
            p1[i] += step;
            double e1 = Etot_fake(p1[0], p1[1], p1[2], p1[3], p1[4], p1[5]);
            param_opt[i].add_point(p0[i], e0, (e1 - e0) / step);
        }
        for (int i = 0; i < 6; i++) {
            params[i] = param_opt[i].next_x();
        }

        double d{0};
        for (int i = 0; i < 6; i++) {
            d += std::abs(param_opt[i].step());
        }
        std::cout << "diff in step=" << d << std::endl;
        if (d < 1e-5) {
            printf("Done in %i iterations!\n", iter);
            break;
        }
    }
    param_opt[0].estimate_x0();
    param_opt[3].estimate_x0();
}

void volume_relaxation(task_t task, cmd_args args, Parameters_input_section& inp)
{
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");

    std::map<double, double> etot;

    auto run_gs = [&etot, &fname, &args, &inp](double s) {
        auto ctx0 = create_sim_ctx(fname, args, inp);
        auto lv = ctx0->unit_cell().lattice_vectors();
        ctx0->unit_cell().set_lattice_vectors(lv * s);
        ctx0->initialize();

        double e = ground_state(*ctx0, task_t::ground_state_new, args, inp, 0);
        
        //double e = Etot_fake(ctx0->unit_cell().lattice_vector(0),
        //                     ctx0->unit_cell().lattice_vector(1),
        //                     ctx0->unit_cell().lattice_vector(2));
        etot[s] = e;

        return e;
    };

    json dict;
    json_output_common(dict);
    dict["task"] = static_cast<int>(task);

    double scale{1};
    double step{0.05};
    double e1 = run_gs(scale);
    double de{0}, de_prev;
    int sgn{1};
    int num_sign_changes{0};
    
    for (int iter = 0; iter < 100; iter++) {
        double e0 = e1;
        e1 = run_gs(scale + sgn * step);
        de_prev = de;
        de = sgn * (e1 - e0) / step;
        
        scale = scale + sgn * step;

        if (iter > 0) {
            if (de_prev * de < 0) {
                step *= 0.55;
                num_sign_changes++;
            } else {
                step *= 1.25;
            }
        }
        sgn = -Utils::sign(de);

        if (etot.size() > 4 && num_sign_changes > 0) {
            printf("converged in %i iterations\n", iter);
            break;
        }
    }
    std::vector<double> x, y;

    for (auto it: etot) {
        x.push_back(it.first);
        y.push_back(it.second);
    }
    Radial_grid scale_steps(x);
    Spline<double> e(scale_steps, y);
    
    double e0{1e100};
    double scale0{0};
    for (int i = 0; i < scale_steps.num_points() - 1; i++) {
        double dx = scale_steps.dx(i) / 1000.0;
        for (int j = 0; j < 1000; j++) {
            if (e(i, dx * j) < e0) {
                e0 = e(i, dx * j);
                scale0 = scale_steps[i] + dx * j;
            }
        }
    }

    dict["etot"] = json::object();
    dict["etot"]["x"] = x;
    dict["etot"]["y"] = y;
    dict["etot"]["scale0"] = scale0;

    if (true) {
        auto ctx0 = create_sim_ctx(fname, args, inp);
        auto lv = ctx0->unit_cell().lattice_vectors();
        ctx0->unit_cell().set_lattice_vectors(lv * scale0);
        dict["unit_cell"] = ctx0->unit_cell().serialize();
        dict["task_status"] = "success";
    } else {
        dict["task_status"] = "failure";
    }

    if (mpi_comm_world().rank() == 0) {
        std::ofstream ofs("relaxed_unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);

        if (args.exist("aiida_output")) {
            std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
            ofs << dict.dump(4);
        }
    }
}

void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));
    /* get the input file name */
    std::string fname = args.value<std::string>("input", "sirius.json");
    /* read json file */
    json dict;
    std::ifstream(fname) >> dict;
    /* read input section */
    Parameters_input_section inp;
    inp.read(dict);

    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args, inp);
        ctx->initialize();
        ground_state(*ctx, task, args, inp, 1);
    }

    if (task == task_t::lattice_relaxation_new) {
        lattice_relaxation(task, args, inp);
    }

    if (task == task_t::volume_relaxation_new || task == task_t::volume_relaxation_descent) {
        volume_relaxation(task, args, inp);
    }

}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--mpi_grid=", "{vector int} MPI grid dimensions");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");

    args.parse_args(argn, argv);

    if (args.exist("help")) {
        printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    run_tasks(args);
    
    sirius::finalize();
    return 0;
}

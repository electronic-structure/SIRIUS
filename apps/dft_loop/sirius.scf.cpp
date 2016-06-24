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

//== void write_json_output(Simulation_context& ctx, DFT_ground_state& gs, bool aiida_output, int result)
//== {
//==     json dict;
//==     json_output_common(dict);
//==     
//==     dict["ground_state"] = gs.serialize();
//==     dict["timers"] = runtime::Timer::serialize();
//==  
//==     if (ctx.comm().rank() == 0) {
//==         std::string fname = std::string("output_") + ctx.start_time_tag() + std::string(".json");
//== 
//==         std::ofstream ofs(fname, std::ofstream::out | std::ofstream::trunc);
//==         ofs << dict.dump(4);
//==         ofs.close();
//==     }
//== 
//==     //== if (ctx.comm().rank() == 0 && aiida_output) {
//==     //==     std::string fname = std::string("output_aiida.json");
//==     //==     JSON_write jw(fname);
//==     //==     if (result >= 0) {
//==     //==         jw.single("status", "converged");
//==     //==         jw.single("num_scf_iterations", result);
//==     //==     } else {
//==     //==         jw.single("status", "unconverged");
//==     //==     }
//== 
//==     //==     jw.single("volume", ctx.unit_cell().omega() * std::pow(au2angs, 3));
//==     //==     jw.single("volume_units", "angstrom^3");
//==     //==     jw.single("energy", etot * ha2ev);
//==     //==     jw.single("energy_units", "eV");
//==     //== }
//== }

Simulation_context* create_sim_ctx(std::string                     fname__,
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
    if (ctx.esm_type() == full_potential_lapwlo) {
        ctx.set_lmax_apw(inp__.lmax_apw_);
        ctx.set_lmax_pot(inp__.lmax_pot_);
        ctx.set_lmax_rho(inp__.lmax_rho_);
    }
    ctx.set_num_mag_dims(inp__.num_mag_dims_);
    ctx.set_auto_rmt(inp__.auto_rmt_);
    ctx.set_core_relativity(inp__.core_relativity_);
    ctx.set_valence_relativity(inp__.valence_relativity_);
    ctx.set_gamma_point(inp__.gamma_point_);

    return ctx_ptr;
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
    
    int result = dft.find(inp.potential_tol_, inp.energy_tol_, inp.num_dft_iter_);
    
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

    //std::cout << ref_lat_param.a << " " << ref_lat_param.b << " " << ref_lat_param.c << " "
    //          << ref_lat_param.alpha << " " << ref_lat_param.beta << " " << ref_lat_param.gamma << std::endl;

    //std::cout << a__ << " " << b__ << " " << c__ << " "
    //          << alpha__ << " " << beta__ << " " << gamma__ << std::endl;

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
        std::unique_ptr<Simulation_context> ctx0(create_sim_ctx(fname, args, inp));
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

    if (task == task_t::volume_relaxation_new) {
        /* test first three points */
        std::vector<double> s0({0.9, 1.0, 1.1});
        for (double s: s0) {
            run_gs(s);
        }
        
        std::vector<double> s1;
        if (etot[s0[1]] < etot[s0[0]] && etot[s0[1]] < etot[s0[2]]) {
            s1 = std::vector<double>({0.95, 1.05});
        }
        if (etot[s0[1]] >= etot[s0[0]]) {
            s1 = std::vector<double>({0.8, 0.85});
        }
        if (etot[s0[1]] >= etot[s0[2]]) {
            s1 = std::vector<double>({1.15, 1.2});
        }

        for (double s: s1) {
            run_gs(s);
        }

        std::vector<double> x;
        std::vector<double> y;

        for (auto it: etot) {
            x.push_back(it.first);
            y.push_back(it.second);
        }
        Radial_grid scale(x);
        Spline<double> e(scale, y);


        //int npt{5};

        //Radial_grid scale(linear_grid, npt, 0.9, 1.1);
        //Spline<double> etot(scale);
        //
        //for (int i = 0; i < npt; i++) {
        //    std::unique_ptr<Simulation_context> ctx0(create_sim_ctx(fname, args, inp));

        //    auto a0 = ctx0->unit_cell().lattice_vector(0) * scale[i];
        //    auto a1 = ctx0->unit_cell().lattice_vector(1) * scale[i];
        //    auto a2 = ctx0->unit_cell().lattice_vector(2) * scale[i];
        //    ctx0->unit_cell().set_lattice_vectors(a0, a1, a2);
        //    ctx0->initialize();

        //    etot[i] = Etot_fake(a0, a1, a2);
        //    //etot[i] = ground_state(*ctx0, task_t::ground_state_new, args, inp, 0);
        //}
        //etot.interpolate();



        double scale0{0};
        int found{0};
        for (int i = 0; i < scale.num_points() - 1; i++) {
            if (e.deriv(1, i) * e.deriv(1, i + 1) < 0) {
                for (int j = 0; j < 10000; j++) {
                    double dx = scale.dx(i) / 10000.0;
                    if (e.deriv(1, i, dx * j) * e.deriv(1, i, dx * (j + 1)) < 0) {
                        if (!found) {
                            scale0 = scale[i] + dx * j;
                        }
                        found++;
                        break;
                    }
                }
            }
        }
        if (found > 1) {
            WARNING("more than one minimum has been found");
        }
        
        dict["etot"] = e.values();
        if (found == 1) {
            std::unique_ptr<Simulation_context> ctx0(create_sim_ctx(fname, args, inp));
            auto a0 = ctx0->unit_cell().lattice_vector(0) * scale0;
            auto a1 = ctx0->unit_cell().lattice_vector(1) * scale0;
            auto a2 = ctx0->unit_cell().lattice_vector(2) * scale0;
            ctx0->unit_cell().set_lattice_vectors(a0, a1, a2);

            dict["unit_cell"] = ctx0->unit_cell().serialize();
            dict["task_status"] = "success";

        } else {
            dict["task_status"] = "failure";
        }

    }

    if (task == task_t::volume_relaxation_descent) {
        double scale = 1.0;
        double step = 0.05;
        double e1 = run_gs(scale);
        double de{0}, de_prev;
        int sgn = 1;
        
        bool found{false};
        for (int iter = 0; iter < 100; iter++) {
            double e0 = e1;
            e1 = run_gs(scale + sgn * step);
            de_prev = de;
            de = sgn * (e1 - e0) / step;
            
            scale = scale + sgn * step;

            if (iter > 0) {
                if (de_prev * de < 0) {
                    step *= 0.5;
                }
            }
            sgn = -Utils::sign(de);

            if (step < 1e-3) {
                found = true;
                printf("converged in %i iterations\n", iter);
                break;
            }
        }

        //== double scale = 1.0;
        //== Parameter_optimization scale_opt(0.01);
        //== 
        //== bool found{false};
        //== for (int iter = 0; iter < 100; iter++) {
        //==     double e0 = run_gs(scale);
        //==     double step = 1e-6;
        //==     double e1 = run_gs(scale + step);
        //==     scale_opt.add_point(scale, e0, (e1 - e0) / step);
        //==     scale = scale_opt.next_x();

        //==     if (scale_opt.step() < 1e-6) {
        //==         found = true;
        //==         printf("converged in %i iterations\n", iter);
        //==         break;
        //==     }
        //== }

        if (found) {
            std::unique_ptr<Simulation_context> ctx0(create_sim_ctx(fname, args, inp));
            auto a0 = ctx0->unit_cell().lattice_vector(0) * scale;
            auto a1 = ctx0->unit_cell().lattice_vector(1) * scale;
            auto a2 = ctx0->unit_cell().lattice_vector(2) * scale;
            ctx0->unit_cell().set_lattice_vectors(a0, a1, a2);

            dict["unit_cell"] = ctx0->unit_cell().serialize();
            dict["task_status"] = "success";
        } else {
            dict["task_status"] = "failure";
        }
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
    /* read input section */
    JSON_tree parser(fname);
    Parameters_input_section inp;
    inp.read(parser);

    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        std::unique_ptr<Simulation_context> ctx(create_sim_ctx(fname, args, inp));
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

    args.parse_args(argn, argv);

    if (args.exist("help")) {
        printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);

    run_tasks(args);
    
    sirius::finalize();
}

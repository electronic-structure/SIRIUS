
#define GET_SIM_CTX(h) auto& sim_ctx = static_cast<utils::any_ptr*>(*h)->get<sirius::Simulation_context>();

/* @fortran begin function void sirius_initialize       Initialize the SIRIUS library.
   @fortran argument in required bool call_mpi_init     If .true. then MPI_Init must be called prior to initialization.
   @fortran end */
void sirius_initialize(bool const* call_mpi_init__)
{
    sirius::initialize(*call_mpi_init__);
}

/* @fortran begin function void sirius_finalize         Shut down the SIRIUS library
   @fortran argument in required bool call_mpi_fin      If .true. then MPI_Finalize must be called after the shutdown.
   @fortran end */
void sirius_finalize(bool const* call_mpi_fin__)
{
    sirius::finalize(*call_mpi_fin__);
}

/* @fortran begin function void sirius_start_timer      Start the timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_start_timer(char const* name__)
{
    std::string name(name__);
    if (!utils::timer::ftimers().count(name)) {
        utils::timer::ftimers().insert(std::make_pair(name, utils::timer(name)));
    } else {
        std::stringstream s;
        s << "timer " << name__ << " is already active";
        TERMINATE(s);
    }
}

/* @fortran begin function void sirius_stop_timer       Stop the running timer.
   @fortran argument in required string name            Timer label.
   @fortran end */
void sirius_stop_timer(char const* name__)
{
    std::string name(name__);
    if (utils::timer::ftimers().count(name)) {
        utils::timer::ftimers().erase(name);
    }
}

/* @fortran begin function bool sirius_context_initialized      Check if the simulation context is initialized. 
   @fortran argument in required void* handler                  Simulation context handler.
   @fortran end */
bool sirius_context_initialized(void* const* handler__)
{
    if (*handler__ == nullptr) {
        return false;
    }
    GET_SIM_CTX(handler__);
    return sim_ctx.initialized();
}

/* @fortran begin function void sirius_create_context_v2         Create context of the simulation.
   @fortran argument out required void* handler                  Simulation context handler.
   @fortran argument in  required int   fcomm                    Entire communicator of the simulation. 
   @fortran end */
void sirius_create_context_v2(void**     handler__,
                              int const* fcomm__)
{
    auto& comm = Communicator::map_fcomm(*fcomm__);
    *handler__ = new utils::any_ptr(new sirius::Simulation_context(comm));
}

/* @fortran begin function void sirius_import_parameters_v2        Import parameters of simulation from a JSON string
   @fortran argument in required void* handler                     Simulation context handler.
   @fortran argument in required string json_str                   JSON string with parameters.
   @fortran end */
void sirius_import_parameters_v2(void* const* handler__,
                                 char  const* str__)
{
    GET_SIM_CTX(handler__);
    sim_ctx.import(std::string(str__));
}

/* @fortran begin function void sirius_set_parameters  Set parameters of the simulation.
   @fortran argument in required void* handler       Simulation context handler
   @fortran argument in optional int lmax_apw        Maximum orbital quantum number for APW functions.
   @fortran argument in optional int lmax_rho        Maximum orbital quantum number for density. 
   @fortran argument in optional int lmax_pot        Maximum orbital quantum number for potential.
   @fortran argument in optional int num_bands       Number of bands.
   @fortran argument in optional int num_mag_dims    Number of magnetic dimensions. 
   @fortran argument in optional double pw_cutoff    Cutoff for G-vectors.
   @fortran argument in optional double gk_cutoff    Cutoff for G+k-vectors.
   @fortran argument in optional double aw_cutoff    This is R_{mt} * gk_cutoff.
   @fortran argument in optional int auto_rmt        Set the automatic search of muffin-tin radii.
   @fortran argument in optional bool gamma_point    True if this is a Gamma-point calculation.
   @fortran argument in optional string valence_rel  Valence relativity treatment.
   @fortran argument in optional string core_rel     Core relativity treatment.
   @fortran end */ 
void sirius_set_parameters(void*  const* handler__,
                           int    const* lmax_apw__,
                           int    const* lmax_rho__,
                           int    const* lmax_pot__,
                           int    const* num_bands__,
                           int    const* num_mag_dims__,
                           double const* pw_cutoff__,
                           double const* gk_cutoff__,
                           double const* aw_cutoff__,
                           int    const* auto_rmt__,
                           bool   const* gamma_point__,
                           char   const* valence_rel__,
                           char   const* core_rel__)
{
    GET_SIM_CTX(handler__);
    if (lmax_apw__ != nullptr) {
        sim_ctx.set_lmax_apw(*lmax_apw__);
    }
    if (lmax_rho__ != nullptr) {
        sim_ctx.set_lmax_rho(*lmax_rho__);
    }
    if (lmax_pot__ != nullptr) {
        sim_ctx.set_lmax_pot(*lmax_pot__);
    }
    if (num_bands__ != nullptr) {
        sim_ctx.num_bands(*num_bands__);
    }
    if (num_mag_dims__ != nullptr) {
        sim_ctx.set_num_mag_dims(*num_mag_dims__);
    }
    if (pw_cutoff__ != nullptr) {
        sim_ctx.set_pw_cutoff(*pw_cutoff__);
    }
    if (gk_cutoff__ != nullptr) {
        sim_ctx.set_gk_cutoff(*gk_cutoff__);
    }
    if (aw_cutoff__ != nullptr) {
        sim_ctx.set_aw_cutoff(*aw_cutoff__);
    }
    if (auto_rmt__ != nullptr) {
        sim_ctx.set_auto_rmt(*auto_rmt__);
    }
    if (gamma_point__ != nullptr) {
        sim_ctx.set_gamma_point(*gamma_point__);
    }
    if (valence_rel__ != nullptr) {
        sim_ctx.set_valence_relativity(valence_rel__);
    }
    if (core_rel__ != nullptr) {
        sim_ctx.set_core_relativity(core_rel__);
    }
}

/* @fortran begin function void sirius_set_lattice_vectors_v2   Set vectors of the unit cell.
   @fortran argument in required void* handler       Simulation context handler
   @fortran argument in required double a1           1st vector
   @fortran argument in required double a2           2nd vector
   @fortran argument in required double a3           3er vector
   @fortran end */
void sirius_set_lattice_vectors_v2(void*  const* handler__,
                                   double const* a1__,
                                   double const* a2__,
                                   double const* a3__)
{
    GET_SIM_CTX(handler__);
    sim_ctx.unit_cell().set_lattice_vectors(vector3d<double>(a1__), vector3d<double>(a2__), vector3d<double>(a3__));
}

/* @fortran begin function void sirius_initialize_context_v2     Initialize simulation context.
   @fortran argument in required void* handler                   Simulation context handler.
   @fortran end */
void sirius_initialize_context_v2(void* const* handler__)
{
    GET_SIM_CTX(handler__)
    sim_ctx.initialize();
}

/* @fortran begin function void sirius_delete_object     Delete any object created by SIRIUS.
   @fortran argument inout required void* handler        Handler of the object.
   @fortran end */ 
void sirius_delete_object(void** handler__)
{
    delete static_cast<utils::any_ptr*>(*handler__);
    *handler__ = nullptr;
}

/* @fortran begin function void sirius_set_periodic_function_ptr   Set pointer to density or megnetization.
   @fortran argument in required void* handler                     Handler of the DFT ground state object.
   @fortran argument in required string label                      Label of the function.
   @fortran argument in required double f_mt                       Pointer to the muffin-tin part of the function.
   @fortran argument in required double f_rg                       Pointer to the regualr-grid part of the function.
   @fortran end */ 
void sirius_set_periodic_function_ptr(void*  const* handler__,
                                      char   const* label__,
                                      double*       f_mt__,
                                      double*       f_rg__)
{
    auto& dft = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();
    std::string label(label__);

    std::map<std::string, sirius::Periodic_function<double>*> func_map = {
        {"rho",  &dft.density().component(0)},
        {"magz", &dft.density().component(1)},
        {"magx", &dft.density().component(2)},
        {"magy", &dft.density().component(3)},
        {"veff", &dft.potential().component(0)},
        {"bz",   &dft.potential().component(1)},
        {"bx",   &dft.potential().component(2)},
        {"by",   &dft.potential().component(3)}
    };

    sirius::Periodic_function<double>* f;
    try {
        f = func_map.at(label);
    } catch(...) {
        std::stringstream s;
        s << "wrong label: " << label;
        TERMINATE(s);
    }

    if (f_mt__) {
        f->set_mt_ptr(f_mt__);
    }
    if (f_rg__) {
        f->set_rg_ptr(f_rg__);
    }
}

/* @fortran begin function void sirius_create_kset_v2      Create k-point set from the list of k-points.
   @fortran argument in  required void*  handler           Simulation context handler.
   @fortran argument out required void*  ks_handler        Handler of the created k-point set.
   @fortran argument in  required int    num_kpoints       Total number of k-points in the set.
   @fortran argument in  required double kpoints           List of k-points in lattice coordinates.
   @fortran argument in  required double kpoint_weights    Weights of k-points.
   @fortran argument in  required bool   init_kset         If .true. k-set will be initialized.
   @fortran end */ 
void sirius_create_kset_v2(void* const*  handler__,
                           void**        ks_handler__,
                           int    const* num_kpoints__,
                           double*       kpoints__,
                           double const* kpoint_weights__,
                           bool   const* init_kset__)
{
    GET_SIM_CTX(handler__);

    mdarray<double, 2> kpoints(kpoints__, 3, *num_kpoints__);

    sirius::K_point_set* new_kset = new sirius::K_point_set(sim_ctx);
    new_kset->add_kpoints(kpoints, kpoint_weights__);
    if (*init_kset__) {
        //std::vector<int> counts;
        new_kset->initialize();
    }

    *ks_handler__ = new utils::any_ptr(new_kset);
}

/* @fortran begin function void sirius_create_ground_state_v2     Create a ground state object.
   @fortran argument in  required void*  ks_handler               Handler of the created k-point set.
   @fortran argument out required void*  gs_handler               Handler of the ground state object.
   @fortran end */ 
void sirius_create_ground_state_v2(void* const* ks_handler__,
                                   void**       gs_handler__)
{
    auto& ks = static_cast<utils::any_ptr*>(*ks_handler__)->get<sirius::K_point_set>();

    *gs_handler__ = new utils::any_ptr(new sirius::DFT_ground_state(ks));
}

/* @fortran begin function void sirius_add_atom_type_v2     Add new atom type to the unit cell.
   @fortran argument in  required void*  handler            Simulation context handler.
   @fortran argument in  required string label              Atom type unique label.
   @fortran argument in  optional string fname              Species file name (in JSON format).
   @fortran argument in  optional int    zn                 Nucleus charge.
   @fortran argument in  optional string symbol             Atomic symbol.
   @fortran argument in  optional double mass               Atomic mass.
   @fortran argument in  optional bool   spin_orbit         True if spin-orbit correction is enabled for this atom type.
   @fortran end */
void sirius_add_atom_type_v2(void*  const* handler__,
                             char   const* label__,
                             char   const* fname__,
                             int    const* zn__,
                             char   const* symbol__,
                             double const* mass__,
                             bool   const* spin_orbit__)
{
    GET_SIM_CTX(handler__);

    std::string label = std::string(label__);
    std::string fname = (fname__ == nullptr) ? std::string("") : std::string(fname__);
    sim_ctx.unit_cell().add_atom_type(label, fname);

    auto& type = sim_ctx.unit_cell().atom_type(label);
    if (zn__ != nullptr) {
        type.set_zn(*zn__);
    }
    if (symbol__ != nullptr) {
        type.set_symbol(std::string(symbol__));
    }
    if (mass__ != nullptr) {
        type.set_mass(*mass__);
    }
    if (spin_orbit__ != nullptr) {
        type.spin_orbit_coupling(*spin_orbit__);
    }
}

/* @fortran begin function void sirius_set_atom_type_radial_grid_v2     Set radial grid of the atom type.
   @fortran argument in  required void*  handler                        Simulation context handler.
   @fortran argument in  required string label                          Atom type label.
   @fortran argument in  required int    num_radial_points              Number of radial grid points.
   @fortran argument in  required double radial_points                  List of radial grid points.
   @fortran end */
void sirius_set_atom_type_radial_grid_v2(void*  const* handler__,
                                         char   const* label__,
                                         int    const* num_radial_points__,
                                         double const* radial_points__)
{
    GET_SIM_CTX(handler__);

    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_radial_grid(*num_radial_points__, radial_points__);
}

/* @fortran begin function void sirius_set_atom_type_hubbard_v2    Set the hubbard correction for the atomic type.
   @fortran argument in  required void*   handler                  Simulation context handler.
   @fortran argument in  required string  label                    Atom type label.
   @fortran argument in  required int     l                        Orbital quantum number.
   @fortran argument in  required int     n                        ?
   @fortran argument in  required double  occ                      Atomic shell occupancy.
   @fortran argument in  required double  U                        Hubbard U parameter.
   @fortran argument in  required double  J                        Exchange J parameter for the full interaction treatment.
   @fortran argument in  required double  alpha                    J_alpha for the simple interaction treatment.
   @fortran argument in  required double  beta                     J_beta for the simple interaction treatment.
   @fortran argument in  required double  J0                       J0 for the simple interaction treatment.
   @fortran end */
void sirius_set_atom_type_hubbard_v2(void*  const* handler__,
                                     char   const* label__,
                                     int    const* l__,
                                     int    const* n__,
                                     double const* occ__,
                                     double const* U__,
                                     double const* J__,
                                     double const* alpha__,
                                     double const* beta__,
                                     double const* J0__)
{
    GET_SIM_CTX(handler__);
    auto& type = sim_ctx.unit_cell().atom_type(std::string(label__));
    type.set_hubbard_correction(true);
    type.set_hubbard_U(*U__ * 0.5);
    type.set_hubbard_J(J__[1] * 0.5);
    type.set_hubbard_alpha(*alpha__);
    type.set_hubbard_beta(*alpha__);
    type.set_hubbard_coefficients(J__);
    type.set_hubbard_J0(*J0__);
    type.set_hubbard_orbital(*n__, *l__, *occ__);
}

/* @fortran begin function void sirius_add_atom_v2      Add atom to the unit cell.
   @fortran argument in  required void*   handler       Simulation context handler.
   @fortran argument in  required string  label         Atom type label.
   @fortran argument in  required double  position      Atom position in lattice coordinates.
   @fortran argument in  optional double  vector_field  Starting magnetization.
   @fortran end */
void sirius_add_atom_v2(void*  const* handler__,
                        char   const* label__,
                        double const* position__,
                        double const* vector_field__)
{
    GET_SIM_CTX(handler__);
    if (vector_field__ != nullptr) {
        sim_ctx.unit_cell().add_atom(std::string(label__), position__, vector_field__);
    } else {
        sim_ctx.unit_cell().add_atom(std::string(label__), position__);
    }
}

/* @fortran begin function void sirius_set_pw_coeffs_v2      Set plane-wave coefficients of a periodic function.
   @fortran argument in  required void*   handler            Ground state handler.
   @fortran argument in  required string  label              Label of the function.
   @fortran argument in  required complex pw_coeffs          Local array of plane-wave coefficients.
   @fortran argument in  optional int     ngv                Local number of G-vectors.
   @fortran end */
void sirius_set_pw_coeffs_v2(void*                const* handler__,
                             char                 const* label__,
                             std::complex<double> const* pw_coeffs__,
                             int                  const* ngv__,
                             int*                        gvl__,
                             int                  const* comm__)
{
    PROFILE("sirius_api::sirius_set_pw_coeffs");

    auto& gs = static_cast<utils::any_ptr*>(*handler__)->get<sirius::DFT_ground_state>();

    std::string label(label__);

    if (gs.ctx().full_potential()) {
        if (label == "veff") {
            gs.potential().set_veff_pw(pw_coeffs__);
        } else if (label == "rm_inv") {
            gs.potential().set_rm_inv_pw(pw_coeffs__);
        } else if (label == "rm2_inv") {
            gs.potential().set_rm2_inv_pw(pw_coeffs__);
        } else {
            TERMINATE("wrong label");
        }
    } else {
        assert(ngv__ != nullptr);
        assert(gvl__ != nullptr);
        assert(comm__ != nullptr);

        Communicator comm(MPI_Comm_f2c(*comm__));
        mdarray<int, 2> gvec(gvl__, 3, *ngv__);

        std::vector<double_complex> v(gs.ctx().gvec().num_gvec(), 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < *ngv__; i++) {
            vector3d<int> G(gvec(0, i), gvec(1, i), gvec(2, i));
            auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
            if (gvc.length() > gs.ctx().pw_cutoff()) {
                continue;
            }
            int ig = gs.ctx().gvec().index_by_gvec(G);
            if (ig >= 0) {
                v[ig] = pw_coeffs__[i];
            } else {
                if (gs.ctx().gamma_point()) {
                    ig = gs.ctx().gvec().index_by_gvec(G * (-1));
                    if (ig == -1) {
                        std::stringstream s;
                        auto gvc = gs.ctx().unit_cell().reciprocal_lattice_vectors() * vector3d<double>(G[0], G[1], G[2]);
                        s << "wrong index of G-vector" << std::endl
                          << "input G-vector: " << G << " (length: " << gvc.length() << " [a.u.^-1])" << std::endl;
                        TERMINATE(s);
                    } else {
                        v[ig] = std::conj(pw_coeffs__[i]);
                    }
                }
            }
        }
        comm.allreduce(v.data(), gs.ctx().gvec().num_gvec());

        // TODO: check if FFT transformation is necessary
        if (label == "rho") {
            gs.density().rho().scatter_f_pw(v);
            gs.density().rho().fft_transform(1);
        } else if (label == "magz") {
            gs.density().magnetization(0).scatter_f_pw(v);
            gs.density().magnetization(0).fft_transform(1);
        } else if (label == "magx") {
            gs.density().magnetization(1).scatter_f_pw(v);
            gs.density().magnetization(1).fft_transform(1);
        } else if (label == "magy") {
            gs.density().magnetization(2).scatter_f_pw(v);
            gs.density().magnetization(2).fft_transform(1);
        } else if (label == "veff") {
            gs.potential().effective_potential()->scatter_f_pw(v);
            gs.potential().effective_potential()->fft_transform(1);
        } else if (label == "bz") {
            gs.potential().effective_magnetic_field(0)->scatter_f_pw(v);
            gs.potential().effective_magnetic_field(0)->fft_transform(1);
        } else if (label == "bx") {
            gs.potential().effective_magnetic_field(1)->scatter_f_pw(v);
            gs.potential().effective_magnetic_field(1)->fft_transform(1);
        } else if (label == "by") {
            gs.potential().effective_magnetic_field(2)->scatter_f_pw(v);
            gs.potential().effective_magnetic_field(2)->fft_transform(1);
        } else if (label == "vxc") {
            gs.potential().xc_potential()->scatter_f_pw(v);
            gs.potential().xc_potential()->fft_transform(1);
        } else if (label == "vloc") {
            gs.potential().local_potential().scatter_f_pw(v);
            gs.potential().local_potential().fft_transform(1);
        } else if (label == "dveff") {
            gs.potential().dveff().scatter_f_pw(v);
        } else {
            std::stringstream s;
            s << "wrong label in sirius_set_pw_coeffs()" << std::endl
              << "  label: " << label;
            TERMINATE(s);
        }
    }
}



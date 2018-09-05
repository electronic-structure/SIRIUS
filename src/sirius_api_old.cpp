//== void sirius_set_free_atom_density(char const* label__,
//==                                   int32_t const* num_points__,
//==                                   double const* dens__)
//== {
//==     auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
//==     type.set_free_atom_density(*num_points__, dens__);
//== }
//==
//== /// Set the atomic level configuration of the atom type.
//== /** With each call to the function new atomic level is added to the list of atomic levels of the atom type.
//==  *
//==  *  \param [in] label unique label of the atom type
//==  *  \param [in] n principal quantum number of the atomic level
//==  *  \param [in] l angular quantum number of the atomic level
//==  *  \param [in] k kappa quantum number of the atomic level
//==  *  \param [in] occupancy occupancy of the atomic level
//==  *  \param [in] core .true. if the atomic level belongs to the core
//==  *
//==  *  Example
//==     \code{.F90}
//==     do is=1,nspecies
//==     do ist=occ1,spnst(is)
//==         call sirius_set_atom_type_configuration(trim(spfname(is)), spn(ist, is), spl(ist, is),&
//==                                                &spk(ist, is), spocc(ist, is),&
//==                                                &spcore(ist, is))
//==       enddo
//==     enddo
//==     \endcode
//==  */
//== void sirius_set_atom_type_configuration(ftn_char    label__,
//==                                         ftn_int*    n__,
//==                                         ftn_int*    l__,
//==                                         ftn_int*    k__,
//==                                         ftn_double* occupancy__,
//==                                         ftn_bool*   core__)
//== {
//==     auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
//==     type.set_configuration(*n__, *l__, *k__, *occupancy__, *core__);
//== }
//==
//==
//== /// Set the table of equivalent atoms.
//== /** \param [in] equivalent_atoms table of equivalent atoms
//==
//==     Equivalent atoms are symmetry related and belong to the same atom symmetry class. If equivalence table is not
//==     provided by user, \a spglib is called. In case of magnetic symmetry \a spglib is of no use and euivalence table
//==     must be provided.
//== */
//== void sirius_set_equivalent_atoms(int32_t* equivalent_atoms__)
//== {
//==     sim_ctx->unit_cell().set_equivalent_atoms(equivalent_atoms__);
//== }
//==
//==
//== void sirius_add_xc_functional(char const* name__)
//== {
//==     assert(name__ != NULL);
//==     sim_ctx->add_xc_functional(name__);
//== }
//==
//== /// Get maximum number of muffin-tin radial points.
//== /** \param [out] max_num_mt_points maximum number of muffin-tin points */
//== void sirius_get_max_num_mt_points(ftn_int* max_num_mt_points__)
//== {
//==     *max_num_mt_points__ = sim_ctx->unit_cell().max_num_mt_points();
//== }
//==
//== /// Get number of muffin-tin radial points for a specific atom type.
//== /** \param [in] label unique label of atom type
//==  *  \param [out] num_mt_points number of muffin-tin points
//==  */
//== void sirius_get_num_mt_points(ftn_char label__,
//==                               ftn_int* num_mt_points__)
//== {
//==     *num_mt_points__ = sim_ctx->unit_cell().atom_type(std::string(label__)).num_mt_points();
//== }
//==
//== void sirius_get_mt_points(ftn_char label__,
//==                           ftn_double* mt_points__)
//== {
//==     auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
//==     for (int i = 0; i < type.num_mt_points(); i++) mt_points__[i] = type.radial_grid(i);
//== }
//==
//== void sirius_get_num_fft_grid_points(int32_t* num_grid_points__)
//== {
//==     *num_grid_points__ = sim_ctx->fft().local_size();
//== }
//==
//== void sirius_get_num_bands(int32_t* num_bands)
//== {
//==     *num_bands = sim_ctx->num_bands();
//== }
//==
//== /// Get number of G-vectors within the plane-wave cutoff
//== void sirius_get_num_gvec(int32_t* num_gvec__)
//== {
//==     *num_gvec__ = sim_ctx->gvec().num_gvec();
//== }
//==
//== void sirius_find_fft_grid_size(ftn_double* cutoff__,
//==                                ftn_double* lat_vec__,
//==                                ftn_int*    grid_size__)
//== {
//==     FFT3D_grid grid(find_translations(*cutoff__, {{lat_vec__[0], lat_vec__[3], lat_vec__[6]},
//==                                                   {lat_vec__[1], lat_vec__[4], lat_vec__[7]},
//==                                                   {lat_vec__[2], lat_vec__[5], lat_vec__[8]}}));
//==     for (int x: {0, 1, 2}) {
//==         grid_size__[x] = grid.size(x);
//==     }
//== }
//==
//== /// Get sizes of FFT grid
//== void sirius_get_fft_grid_size(ftn_int* grid_size__)
//== {
//==     for (int x: {0, 1, 2}) {
//==         grid_size__[x] = sim_ctx->fft().size(x);
//==     }
//== }
//==
//== /// Get lower and upper limits of the FFT grid dimension
//== /** \param [in] d index of dimension (1,2, or 3)
//==  *  \param [out] lower lower (most negative) value
//==  *  \param [out] upper upper (most positive) value
//==  *
//==  *  Example:
//==     \code{.F90}
//==     do i=1,3
//==       call sirius_get_fft_grid_limits(i,intgv(i,1),intgv(i,2))
//==     enddo
//==     \endcode
//==  */
//== void sirius_get_fft_grid_limits(int32_t const* d, int32_t* lower, int32_t* upper)
//== {
//==     assert((*d >= 1) && (*d <= 3));
//==     *lower = sim_ctx->fft().limits(*d - 1).first;
//==     *upper = sim_ctx->fft().limits(*d - 1).second;
//== }
//==
//== /// Get mapping between G-vector index and FFT index
//== void sirius_get_fft_index(int32_t* fft_index__)
//== {
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         auto G = sim_ctx->gvec().gvec(ig);
//==         fft_index__[ig] = sim_ctx->fft().index_by_freq(G[0], G[1], G[2]) + 1;
//==     }
//== }
//==
//== /// Get list of G-vectors in fractional corrdinates
//== void sirius_get_gvec(int32_t* gvec__)
//== {
//==     mdarray<int, 2> gvec(gvec__, 3, sim_ctx->gvec().num_gvec());
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         auto gv = sim_ctx->gvec().gvec(ig);
//==         for (int x: {0, 1, 2}) {
//==             gvec(x, ig) = gv[x];
//==         }
//==     }
//== }
//==
//== /// Get list of G-vectors in Cartesian coordinates
//== void sirius_get_gvec_cart(double* gvec_cart__)
//== {
//==     mdarray<double, 2> gvec_cart(gvec_cart__, 3, sim_ctx->gvec().num_gvec());
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         auto gvc = sim_ctx->gvec().gvec_cart(ig);
//==         for (int x: {0, 1, 2}) {
//==             gvec_cart(x, ig) = gvc[x];
//==         }
//==     }
//== }
//==
//== /// Get lengh of G-vectors
//== void sirius_get_gvec_len(double* gvec_len__)
//== {
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         gvec_len__[ig] = sim_ctx->gvec().gvec_len(ig);
//==     }
//== }
//==
//== void sirius_get_index_by_gvec(int32_t* index_by_gvec__)
//== {
//==     auto d0 = sim_ctx->fft().limits(0);
//==     auto d1 = sim_ctx->fft().limits(1);
//==     auto d2 = sim_ctx->fft().limits(2);
//==
//==     mdarray<int, 3> index_by_gvec(index_by_gvec__, d0, d1, d2);
//==     std::fill(index_by_gvec.at<CPU>(), index_by_gvec.at<CPU>() + index_by_gvec.size(), -1);
//==
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         auto G = sim_ctx->gvec().gvec(ig);
//==         index_by_gvec(G[0], G[1], G[2]) = ig + 1;
//==     }
//== }
//==
//== /// Get Ylm spherical harmonics of G-vectors.
//== void sirius_get_gvec_ylm(double_complex* gvec_ylm__, int* ld__, int* lmax__)
//== {
//==     TERMINATE("fix this");
//==
//==     //==mdarray<double_complex, 2> gvec_ylm(gvec_ylm__, *ld__, sim_ctx->reciprocal_lattice()->num_gvec());
//==     //==// TODO: can be parallelized
//==     //==for (int ig = 0; ig < sim_ctx->reciprocal_lattice()->num_gvec(); ig++)
//==     //=={
//==     //==    sim_ctx->reciprocal_lattice()->gvec_ylm_array<global>(ig, &gvec_ylm(0, ig), *lmax__);
//==     //==}
//== }
//==
//== void sirius_get_gvec_phase_factors(double_complex* sfacg__)
//== {
//==     TERMINATE("fix this");
//==     //mdarray<double_complex, 2> sfacg(sfacg__, sim_ctx->fft().num_gvec(), sim_ctx->unit_cell().num_atoms());
//==     //for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
//==     //{
//==     //    for (int ig = 0; ig < sim_ctx->fft().num_gvec(); ig++)
//==     //        sfacg(ig, ia) = sim_ctx->reciprocal_lattice()->gvec_phase_factor(ig, ia);
//==     //}
//== }
//==
//== void sirius_get_step_function(double_complex* cfunig__, double* cfunir__)
//== {
//==     for (int i = 0; i < sim_ctx->fft().local_size(); i++) {
//==         cfunir__[i] = sim_ctx->step_function().theta_r(i);
//==     }
//==     for (int ig = 0; ig < sim_ctx->gvec().num_gvec(); ig++) {
//==         cfunig__[ig] = sim_ctx->step_function().theta_pw(ig);
//==     }
//== }
//==
//== /// Get the total number of electrons
//== void sirius_get_num_electrons(double* num_electrons__)
//== {
//==     *num_electrons__ = sim_ctx->unit_cell().num_electrons();
//== }
//==
//== /// Get the number of valence electrons
//== void sirius_get_num_valence_electrons(double* num_valence_electrons__)
//== {
//==     *num_valence_electrons__ = sim_ctx->unit_cell().num_valence_electrons();
//== }
//==
//== /// Get the number of core electrons
//== void sirius_get_num_core_electrons(double* num_core_electrons__)
//== {
//==     *num_core_electrons__ = sim_ctx->unit_cell().num_core_electrons();
//== }
//==
//== void sirius_generate_valence_density(int32_t* kset_id__)
//== {
//==     dft_ground_state->density().generate_valence(*kset_list[*kset_id__]);
//==     if (sim_ctx->full_potential()) {
//==         /* only PW coeffs have been generated; transfrom them to real space */
//==         dft_ground_state->density().fft_transform(1);
//==         /* MT part was calculated for local number of atoms; synchronize to global array */
//==         dft_ground_state->density().rho().sync_mt();
//==         for (int j = 0; j < sim_ctx->num_mag_dims(); j++) {
//==             dft_ground_state->density().magnetization(j).sync_mt();
//==         }
//==     }
//== }
//==
//== void sirius_augment_density(int32_t* kset_id__)
//== {
//==     dft_ground_state->density().augment(*kset_list[*kset_id__]);
//== }
//==
//==
//== void sirius_find_band_occupancies(int32_t* kset_id__)
//== {
//==     kset_list[*kset_id__]->find_band_occupancies();
//== }
//==
//==
//== void sirius_print_timers(void)
//== {
//==     if (sim_ctx->comm().rank() == 0) {
//==         utils::timer::print();
//==     }
//== }
//==
//== void sirius_save_potential(void)
//== {
//==     dft_ground_state->potential().save();
//== }
//==
//== void sirius_save_density(void)
//== {
//==     dft_ground_state->density().save();
//== }
//==
//== void sirius_load_potential(void)
//== {
//==     dft_ground_state->potential().load();
//== }
//==
//== void sirius_load_density(void)
//== {
//==     dft_ground_state->density().load();
//== }
//==
//==
//== void FORTRAN(sirius_plot_potential)(void)
//== {
//==     int N{10000};
//==
//==     dft_ground_state->potential().effective_potential().fft_transform(-1);
//==
//==     std::vector<double> p(N);
//==     std::vector<double> x(N);
//==
//==     vector3d<double> vf1({0.1, 0.1, 0.1});
//==     vector3d<double> vf2({0.4, 0.4, 0.4});
//==
//==     #pragma omp parallel for default(shared)
//==     for (int i = 0; i < N; i++) {
//==         double t = double(i) / (N - 1);
//==         auto vf = vf1 + (vf2 - vf1) * t;
//==
//==         auto vc = sim_ctx->unit_cell().get_cartesian_coordinates(vf);
//==         p[i] = dft_ground_state->potential().effective_potential().value(vc);
//==         x[i] = vc.length();
//==     }
//==
//==     FILE* fout = fopen("potential.dat", "w");
//==     for (int i = 0; i < N; i++) {
//==         fprintf(fout, "%.12f %.12f\n", x[i] - x[0], p[i]);
//==     }
//==     fclose(fout);
//== }
//==
//== void sirius_write_json_output(void)
//== {
//==     json dict;
//==     dict["git_hash"] = git_hash;
//==     dict["build_date"] = build_date;
//==     dict["comm_world_size"] = Communicator::world().size();
//==     dict["threads_per_rank"] = omp_get_max_threads();
//==     dict["ground_state"] = dft_ground_state->serialize();
//==     dict["timers"] = utils::timer::serialize_timers();
//==
//==     if (Communicator::world().rank() == 0) {
//==         std::ofstream ofs(std::string("output_") + sim_ctx->start_time_tag() + std::string(".json"),
//==                           std::ofstream::out | std::ofstream::trunc);
//==         ofs << dict.dump(4);
//==     }
//== }
//==
//== void FORTRAN(sirius_get_occupation_matrix)(int32_t* atom_id, double_complex* occupation_matrix)
//== {
//==     int ia = *atom_id - 1;
//==     sim_ctx->unit_cell().atom(ia).get_occupation_matrix(occupation_matrix);
//== }
//==
//== void FORTRAN(sirius_set_uj_correction_matrix)(int32_t* atom_id, int32_t* l, double_complex* uj_correction_matrix)
//== {
//==     int ia = *atom_id - 1;
//==     sim_ctx->unit_cell().atom(ia).set_uj_correction_matrix(*l, uj_correction_matrix);
//== }
//==
//== void FORTRAN(sirius_set_so_correction)(int32_t* so_correction)
//== {
//==     if (*so_correction != 0) {
//==         sim_ctx->set_so_correction(true);
//==     } else {
//==         sim_ctx->set_so_correction(false);
//==     }
//== }
//==
//== void sirius_add_atom_type_aw_descriptor(char const* label__,
//==                                         int32_t const* n__,
//==                                         int32_t const* l__,
//==                                         double const* enu__,
//==                                         int32_t const* dme__,
//==                                         int32_t const* auto_enu__)
//== {
//==     auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
//==     type.add_aw_descriptor(*n__, *l__, *enu__, *dme__, *auto_enu__);
//== }
//==
//== void sirius_add_atom_type_lo_descriptor(char const* label__,
//==                                         int32_t const* ilo__,
//==                                         int32_t const* n__,
//==                                         int32_t const* l__,
//==                                         double const* enu__,
//==                                         int32_t const* dme__,
//==                                         int32_t* auto_enu__)
//== {
//==     std::string label(label__);
//==     auto& type = sim_ctx->unit_cell().atom_type(std::string(label__));
//==     type.add_lo_descriptor(*ilo__ - 1, *n__, *l__, *enu__, *dme__, *auto_enu__);
//== }
//==
//== void sirius_set_aw_enu(int32_t const* ia__,
//==                        int32_t const* l__,
//==                        int32_t const* order__,
//==                        double const* enu__)
//== {
//==     sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_aw_enu(*l__, *order__ - 1, *enu__);
//== }
//==
//== void sirius_get_aw_enu(int32_t const* ia__,
//==                        int32_t const* l__,
//==                        int32_t const* order__,
//==                        double* enu__)
//== {
//==     *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_aw_enu(*l__, *order__ - 1);
//== }
//==
//== void sirius_set_lo_enu(int32_t const* ia__,
//==                        int32_t const* idxlo__,
//==                        int32_t const* order__,
//==                        double const* enu__)
//== {
//==     sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_lo_enu(*idxlo__ - 1, *order__ - 1, *enu__);
//== }
//==
//== void sirius_get_lo_enu(int32_t const* ia__,
//==                        int32_t const* idxlo__,
//==                        int32_t const* order__,
//==                        double* enu__)
//== {
//==     *enu__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().get_lo_enu(*idxlo__ - 1, *order__ - 1);
//== }
//==
//== void sirius_get_local_num_kpoints(int32_t* kset_id, int32_t* nkpt_loc)
//== {
//==     *nkpt_loc = (int)kset_list[*kset_id]->spl_num_kpoints().local_size();
//== }
//==
//== void sirius_get_local_kpoint_rank_and_offset(int32_t* kset_id, int32_t* ik, int32_t* rank, int32_t* ikloc)
//== {
//==     *rank = kset_list[*kset_id]->spl_num_kpoints().local_rank(*ik - 1);
//==     *ikloc = (int)kset_list[*kset_id]->spl_num_kpoints().local_index(*ik - 1) + 1;
//== }
//==
//== void sirius_get_global_kpoint_index(int32_t* kset_id, int32_t* ikloc, int32_t* ik)
//== {
//==     *ik = kset_list[*kset_id]->spl_num_kpoints(*ikloc - 1) + 1; // Fortran counts from 1
//== }
//==
//== /// Generate radial functions (both aw and lo)
//== void sirius_generate_radial_functions()
//== {
//==     sim_ctx->unit_cell().generate_radial_functions();
//== }
//==
//== /// Generate radial integrals
//== void sirius_generate_radial_integrals()
//== {
//==     sim_ctx->unit_cell().generate_radial_integrals();
//== }
//==
//== void sirius_get_symmetry_classes(int32_t* ncls, int32_t* icls_by_ia)
//== {
//==     *ncls = sim_ctx->unit_cell().num_atom_symmetry_classes();
//==
//==     for (int ic = 0; ic < sim_ctx->unit_cell().num_atom_symmetry_classes(); ic++)
//==     {
//==         for (int i = 0; i < sim_ctx->unit_cell().atom_symmetry_class(ic).num_atoms(); i++)
//==             icls_by_ia[sim_ctx->unit_cell().atom_symmetry_class(ic).atom_id(i)] = ic + 1; // Fortran counts from 1
//==     }
//== }
//==
//== void sirius_get_max_mt_radial_basis_size(int32_t* max_mt_radial_basis_size)
//== {
//==     *max_mt_radial_basis_size = sim_ctx->unit_cell().max_mt_radial_basis_size();
//== }
//==
//== void sirius_get_radial_functions(double* radial_functions__)
//== {
//==     mdarray<double, 3> radial_functions(radial_functions__,
//==                                         sim_ctx->unit_cell().max_num_mt_points(),
//==                                         sim_ctx->unit_cell().max_mt_radial_basis_size(),
//==                                         sim_ctx->unit_cell().num_atom_symmetry_classes());
//==     radial_functions.zero();
//==
//==     for (int ic = 0; ic < sim_ctx->unit_cell().num_atom_symmetry_classes(); ic++)
//==     {
//==         for (int idxrf = 0; idxrf < sim_ctx->unit_cell().atom_symmetry_class(ic).atom_type().mt_radial_basis_size(); idxrf++)
//==         {
//==             for (int ir = 0; ir < sim_ctx->unit_cell().atom_symmetry_class(ic).atom_type().num_mt_points(); ir++)
//==                 radial_functions(ir, idxrf, ic) = sim_ctx->unit_cell().atom_symmetry_class(ic).radial_function(ir, idxrf);
//==         }
//==     }
//== }
//==
//== void sirius_get_max_mt_basis_size(int32_t* max_mt_basis_size)
//== {
//==     *max_mt_basis_size = sim_ctx->unit_cell().max_mt_basis_size();
//== }
//==
//== void sirius_get_basis_functions_index(int32_t* mt_basis_size, int32_t* offset_wf, int32_t* indexb__)
//== {
//==     mdarray<int, 3> indexb(indexb__, 4, sim_ctx->unit_cell().max_mt_basis_size(), sim_ctx->unit_cell().num_atoms());
//==
//==     for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
//==     {
//==         mt_basis_size[ia] = sim_ctx->unit_cell().atom(ia).type().mt_basis_size();
//==         offset_wf[ia] = sim_ctx->unit_cell().atom(ia).offset_mt_coeffs();
//==
//==         for (int j = 0; j < sim_ctx->unit_cell().atom(ia).type().mt_basis_size(); j++)
//==         {
//==             indexb(0, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).l;
//==             indexb(1, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).lm + 1; // Fortran counts from 1
//==             indexb(2, j, ia) = sim_ctx->unit_cell().atom(ia).type().indexb(j).idxrf + 1; // Fortran counts from 1
//==         }
//==     }
//== }
//==
//== /// Get number of G+k vectors for a given k-point in the set
//== void sirius_get_num_gkvec(ftn_int* kset_id__,
//==                           ftn_int* ik__,
//==                           ftn_int* num_gkvec__)
//== {
//==     auto ks = kset_list[*kset_id__];
//==     auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//==     /* get rank that stores a given k-point */
//==     int rank = ks->spl_num_kpoints().local_rank(*ik__ - 1);
//==     auto& comm_k = sim_ctx->comm_k();
//==     if (rank == comm_k.rank()) {
//==         *num_gkvec__ = kp->num_gkvec();
//==     }
//==     comm_k.bcast(num_gkvec__, 1, rank);
//== }
//==
//== /// Get maximum number of G+k vectors across all k-points in the set
//== void sirius_get_max_num_gkvec(ftn_int* kset_id__,
//==                               ftn_int* max_num_gkvec__)
//== {
//==     *max_num_gkvec__ = kset_list[*kset_id__]->max_num_gkvec();
//== }
//==
//== /// Get all G+k vector related arrays
//== void sirius_get_gkvec_arrays(ftn_int*    kset_id__,
//==                              ftn_int*    ik__,
//==                              ftn_int*    num_gkvec__,
//==                              ftn_int*    gvec_index__,
//==                              ftn_double* gkvec__,
//==                              ftn_double* gkvec_cart__,
//==                              ftn_double* gkvec_len,
//==                              ftn_double* gkvec_tp__)
//== {
//==
//==     auto ks = kset_list[*kset_id__];
//==     auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//==
//==     /* get rank that stores a given k-point */
//==     int rank = ks->spl_num_kpoints().local_rank(*ik__ - 1);
//==
//==     auto& comm_k = sim_ctx->comm_k();
//==
//==     if (rank == comm_k.rank()) {
//==         *num_gkvec__ = kp->num_gkvec();
//==         mdarray<double, 2> gkvec(gkvec__, 3, kp->num_gkvec());
//==         mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
//==         mdarray<double, 2> gkvec_tp(gkvec_tp__, 2, kp->num_gkvec());
//==
//==         for (int igk = 0; igk < kp->num_gkvec(); igk++) {
//==             auto gkc = kp->gkvec().gkvec_cart(igk);
//==             auto G = kp->gkvec().gvec(igk);
//==
//==             gvec_index__[igk] = sim_ctx->gvec().index_by_gvec(G) + 1; // Fortran counts from 1
//==             for (int x: {0, 1, 2}) {
//==                 gkvec(x, igk) = kp->gkvec().gkvec(igk)[x];
//==                 gkvec_cart(x, igk) = gkc[x];
//==             }
//==             auto rtp = sirius::SHT::spherical_coordinates(gkc);
//==             gkvec_len[igk] = rtp[0];
//==             gkvec_tp(0, igk) = rtp[1];
//==             gkvec_tp(1, igk) = rtp[2];
//==         }
//==     }
//==     comm_k.bcast(num_gkvec__,  1,                rank);
//==     comm_k.bcast(gvec_index__, *num_gkvec__,     rank);
//==     comm_k.bcast(gkvec__,      *num_gkvec__ * 3, rank);
//==     comm_k.bcast(gkvec_cart__, *num_gkvec__ * 3, rank);
//==     comm_k.bcast(gkvec_len,    *num_gkvec__,     rank);
//==     comm_k.bcast(gkvec_tp__,   *num_gkvec__ * 2, rank);
//== }
//==
//== void sirius_get_matching_coefficients(int32_t const* kset_id__,
//==                                       int32_t const* ik__,
//==                                       double_complex* apwalm__,
//==                                       int32_t const* ngkmax__,
//==                                       int32_t const* apwordmax__)
//== {
//==
//==     TERMINATE_NOT_IMPLEMENTED;
//==
//==     //int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
//==     //
//==     //if (rank == sim_ctx->mpi_grid().coordinate(0))
//==     //{
//==     //    auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//==     //
//==     //    mdarray<double_complex, 4> apwalm(apwalm__, *ngkmax__, *apwordmax__, sim_ctx->lmmax_apw(),
//==     //                                      sim_ctx->unit_cell().num_atoms());
//==
//==
//==     //    dmatrix<double_complex> alm(kp->num_gkvec_row(), sim_ctx->unit_cell().mt_aw_basis_size(), *blacs_grid, sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
//==     //    kp->alm_coeffs_row()->generate<true>(alm);
//==
//==     //    for (int i = 0; i < sim_ctx->unit_cell().mt_aw_basis_size(); i++)
//==     //    {
//==     //        int ia = sim_ctx->unit_cell().mt_aw_basis_descriptor(i).ia;
//==     //        int xi = sim_ctx->unit_cell().mt_aw_basis_descriptor(i).xi;
//==     //
//==     //        int lm = sim_ctx->unit_cell().atom(ia).type().indexb(xi).lm;
//==     //        int order = sim_ctx->unit_cell().atom(ia).type().indexb(xi).order;
//==
//==     //        for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
//==     //        {
//==     //            int igk = kp->gklo_basis_descriptor_row(igkloc).igk;
//==     //            apwalm(igk, order, lm, ia) = alm(igkloc, i);
//==     //        }
//==     //    }
//==     //    //== for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++)
//==     //    //== {
//==     //    //==     Platform::allreduce(&apwalm(0, 0, 0, ia), (int)(apwalm.size(0) * apwalm.size(1) * apwalm.size(2)),
//==     //    //==                         sim_ctx->mpi_grid().communicator(1 << _dim_row_));
//==     //    //== }
//==     //}
//== }
//==
//== ///// Get first-variational matrices of Hamiltonian and overlap
//== ///** Radial integrals and plane-wave coefficients of the interstitial potential must be calculated prior to
//== // *  Hamiltonian and overlap matrix construction.
//== // */
//== //void sirius_get_fv_h_o(int32_t const* kset_id__,
//== //                       int32_t const* ik__,
//== //                       int32_t const* size__,
//== //                       double_complex* h__,
//== //                       double_complex* o__)
//== //{
//== //    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
//== //
//== //    if (rank == sim_ctx->mpi_grid().coordinate(0))
//== //    {
//== //        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//== //
//== //        if (*size__ != kp->gklo_basis_size())
//== //        {
//== //            TERMINATE("wrong matrix size");
//== //        }
//== //
//== //        dmatrix<double_complex> h(h__, kp->gklo_basis_size(), kp->gklo_basis_size(), sim_ctx->blacs_grid(), sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
//== //        dmatrix<double_complex> o(o__, kp->gklo_basis_size(), kp->gklo_basis_size(), sim_ctx->blacs_grid(), sim_ctx->cyclic_block_size(), sim_ctx->cyclic_block_size());
//== //        dft_ground_state->band().set_fv_h_o<CPU, electronic_structure_method_t::full_potential_lapwlo>(kp, *potential, h, o);
//== //    }
//== //}
//==
//== //void sirius_solve_fv(int32_t const* kset_id__,
//== //                     int32_t const* ik__,
//== //                     double_complex* h__,
//== //                     double_complex* o__,
//== //                     double* eval__,
//== //                     double_complex* evec__,
//== //                     int32_t const* evec_ld__)
//== //{
//== //    int rank = kset_list[*kset_id__]->spl_num_kpoints().local_rank(*ik__ - 1);
//== //
//== //    if (rank == sim_ctx->mpi_grid().coordinate(0))
//== //    {
//== //        auto kp = (*kset_list[*kset_id__])[*ik__ - 1];
//== //
//== //        dft_ground_state->band().gen_evp_solver().solve(kp->gklo_basis_size(),
//== //                                                        sim_ctx->num_fv_states(),
//== //                                                        h__,
//== //                                                        kp->gklo_basis_size_row(),
//== //                                                        o__,
//== //                                                        kp->gklo_basis_size_row(),
//== //                                                        eval__,
//== //                                                        evec__,
//== //                                                        *evec_ld__,
//== //                                                        kp->gklo_basis_size_row(),
//== //                                                        kp->gklo_basis_size_col());
//== //    }
//== //}
//==
//== ///// Get the total size of wave-function (number of mt coefficients + number of G+k coefficients)
//== //void sirius_get_mtgk_size(int32_t* kset_id, int32_t* ik, int32_t* mtgk_size)
//== //{
//== //    *mtgk_size = (*kset_list[*kset_id])[*ik - 1]->wf_size();
//== //}
//==
//== void sirius_get_spinor_wave_functions(int32_t* kset_id, int32_t* ik, double_complex* spinor_wave_functions__)
//== {
//==     TERMINATE("fix this for distributed WF storage");
//==     //== assert(sim_ctx->num_bands() == (int)sim_ctx->spl_spinor_wf().local_size());
//==
//==     //== sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
//==     //==
//==     //== mdarray<double_complex, 3> spinor_wave_functions(spinor_wave_functions__, kp->wf_size(), sim_ctx->num_spins(),
//==     //==                                             sim_ctx->spl_spinor_wf().local_size());
//==
//==     //== for (int j = 0; j < (int)sim_ctx->spl_spinor_wf().local_size(); j++)
//==     //== {
//==     //==     memcpy(&spinor_wave_functions(0, 0, j), &kp->spinor_wave_function(0, 0, j),
//==     //==            kp->wf_size() * sim_ctx->num_spins() * sizeof(double_complex));
//==     //== }
//== }
//==
//== //== void FORTRAN(sirius_apply_step_function_gk)(int32_t* kset_id, int32_t* ik, double_complex* wf__)
//== //== {
//== //==     int thread_id = Platform::thread_id();
//== //==
//== //==     sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
//== //==     int num_gkvec = kp->num_gkvec();
//== //==
//== //==     sim_ctx->reciprocal_lattice()->fft().input(num_gkvec, kp->fft_index(), wf__, thread_id);
//== //==     sim_ctx->reciprocal_lattice()->fft().transform(1, thread_id);
//== //==     for (int ir = 0; ir < sim_ctx->reciprocal_lattice()->fft().size(); ir++)
//== //==         sim_ctx->reciprocal_lattice()->fft().buffer(ir, thread_id) *= sim_ctx->step_function()->theta_it(ir);
//== //==
//== //==     sim_ctx->reciprocal_lattice()->fft().transform(-1, thread_id);
//== //==     sim_ctx->reciprocal_lattice()->fft().output(num_gkvec, kp->fft_index(), wf__, thread_id);
//== //== }
//==
//== /// Get Cartesian coordinates of G+k vectors
//== void sirius_get_gkvec_cart(int32_t* kset_id, int32_t* ik, double* gkvec_cart__)
//== {
//==     sirius::K_point* kp = (*kset_list[*kset_id])[*ik - 1];
//==     mdarray<double, 2> gkvec_cart(gkvec_cart__, 3, kp->num_gkvec());
//==
//==     for (int igk = 0; igk < kp->num_gkvec(); igk++)
//==     {
//==         for (int x = 0; x < 3; x++) gkvec_cart(x, igk) = kp->gkvec().gkvec(igk)[x]; //kp->gkvec<cartesian>(igk)[x];
//==     }
//== }
//==
//==
//== /// Generate XC potential and magnetic field
//== void sirius_generate_xc_potential(ftn_double* vxcmt__,
//==                                   ftn_double* vxcit__,
//==                                   ftn_double* bxcmt__,
//==                                   ftn_double* bxcit__)
//== {
//==
//==     dft_ground_state->potential().xc(dft_ground_state->density());
//==
//==     dft_ground_state->potential().xc_potential().copy_to_global_ptr(vxcmt__, vxcit__);
//==
//==     if (sim_ctx->num_mag_dims() == 0) {
//==         return;
//==     }
//==     assert(sim_ctx->num_spins() == 2);
//==
//==     /* set temporary array wrapper */
//==     mdarray<double, 4> bxcmt(bxcmt__, sim_ctx->lmmax_pot(), sim_ctx->unit_cell().max_num_mt_points(),
//==                              sim_ctx->unit_cell().num_atoms(), sim_ctx->num_mag_dims());
//==     mdarray<double, 2> bxcit(bxcit__, sim_ctx->fft().local_size(), sim_ctx->num_mag_dims());
//==
//==     if (sim_ctx->num_mag_dims() == 1) {
//==         /* z component */
//==         dft_ground_state->potential().effective_magnetic_field(0).copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
//==     } else {
//==         /* z component */
//==         dft_ground_state->potential().effective_magnetic_field(0).copy_to_global_ptr(&bxcmt(0, 0, 0, 2), &bxcit(0, 2));
//==         /* x component */
//==         dft_ground_state->potential().effective_magnetic_field(1).copy_to_global_ptr(&bxcmt(0, 0, 0, 0), &bxcit(0, 0));
//==         /* y component */
//==         dft_ground_state->potential().effective_magnetic_field(2).copy_to_global_ptr(&bxcmt(0, 0, 0, 1), &bxcit(0, 1));
//==     }
//== }
//==
//== void sirius_generate_rho_multipole_moments(ftn_int*            lmmax__,
//==                                            ftn_double_complex* qmt__)
//== {
//==     mdarray<ftn_double_complex, 2> qmt(qmt__, *lmmax__, sim_ctx->unit_cell().num_atoms());
//==     qmt.zero();
//==
//==     int lmmax = std::min(*lmmax__, sim_ctx->lmmax_rho());
//==
//==     auto l_by_lm = Utils::l_by_lm(Utils::lmax_by_lmmax(lmmax));
//==
//==     for (int ialoc = 0; ialoc < sim_ctx->unit_cell().spl_num_atoms().local_size(); ialoc++) {
//==         int ia = sim_ctx->unit_cell().spl_num_atoms(ialoc);
//==         std::vector<double> tmp(lmmax);
//==         for (int lm = 0; lm < lmmax; lm++) {
//==             int l = l_by_lm[lm];
//==             auto s = dft_ground_state->density().rho().f_mt(ialoc).component(lm);
//==             tmp[lm] = s.integrate(l + 2);
//==         }
//==         sirius::SHT::convert(Utils::lmax_by_lmmax(lmmax), tmp.data(), &qmt(0, ia));
//==         qmt(0, ia) -= sim_ctx->unit_cell().atom(ia).zn() * y00;
//==     }
//==     sim_ctx->comm().allreduce(&qmt(0, 0), static_cast<int>(qmt.size()));
//== }
//==
//== void sirius_generate_coulomb_potential_mt(ftn_int*            ia__,
//==                                           ftn_int*            lmmax_rho__,
//==                                           ftn_double_complex* rho__,
//==                                           ftn_int*            lmmax_pot__,
//==                                           ftn_double_complex* vmt__)
//== {
//==     auto& atom = sim_ctx->unit_cell().atom(*ia__ - 1);
//==
//==     sirius::Spheric_function<function_domain_t::spectral, double_complex> rho(rho__, *lmmax_rho__, atom.radial_grid());
//==     sirius::Spheric_function<function_domain_t::spectral, double_complex> vmt(vmt__, *lmmax_pot__, atom.radial_grid());
//==     dft_ground_state->potential().poisson_vmt<true>(atom, rho, vmt);
//== }
//==
//== void sirius_generate_coulomb_potential(ftn_double* vclmt__,
//==                                        ftn_double* vclit__)
//== {
//==     dft_ground_state->density().rho().fft_transform(-1);
//==     dft_ground_state->potential().poisson(dft_ground_state->density().rho());
//==     dft_ground_state->potential().hartree_potential().copy_to_global_ptr(vclmt__, vclit__);
//== }
//==
//== void sirius_get_vha_el(ftn_double* vha_el__)
//== {
//==     for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++) {
//==         vha_el__[ia] = dft_ground_state->potential().vha_el(ia);
//==     }
//== }
//==
//== void sirius_update_atomic_potential()
//== {
//==     dft_ground_state->potential().update_atomic_potential();
//== }
//==
//== void sirius_radial_solver(ftn_char    type__,
//==                           ftn_int*    zn__,
//==                           ftn_int*    dme__,
//==                           ftn_int*    l__,
//==                           ftn_int*    k__,
//==                           ftn_double* enu__,
//==                           ftn_int*    nr__,
//==                           ftn_double* r__,
//==                           ftn_double* v__,
//==                           ftn_int*    nn__,
//==                           ftn_double* p0__,
//==                           ftn_double* p1__,
//==                           ftn_double* q0__,
//==                           ftn_double* q1__,
//==                           ftn_int     type_len__)
//== {
//==
//==     std::string type(type__, type_len__);
//==     if (type != "none") {
//==         TERMINATE_NOT_IMPLEMENTED;
//==     }
//==
//==     relativity_t rel = relativity_t::none;
//==
//==     sirius::Radial_grid_ext<double> rgrid(*nr__, r__);
//==     std::vector<double> v(v__, v__ + rgrid.num_points());
//==     sirius::Radial_solver solver(*zn__, v, rgrid);
//==
//==     auto result = solver.solve(rel, *dme__, *l__, *k__, *enu__);
//==
//==     *nn__ = std::get<0>(result);
//==     std::memcpy(p0__, std::get<1>(result).data(), rgrid.num_points() * sizeof(double));
//==     std::memcpy(p1__, std::get<2>(result).data(), rgrid.num_points() * sizeof(double));
//==     std::memcpy(q0__, std::get<3>(result).data(), rgrid.num_points() * sizeof(double));
//==     std::memcpy(q1__, std::get<4>(result).data(), rgrid.num_points() * sizeof(double));
//== }
//==
//== void sirius_get_aw_radial_function(ftn_int*    ia__,
//==                                    ftn_int*    l__,
//==                                    ftn_int*    io__,
//==                                    ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int io = *io__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_l_order(*l__, io);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
//==     }
//== }
//==
//== void sirius_set_aw_radial_function(ftn_int*    ia__,
//==                                    ftn_int*    l__,
//==                                    ftn_int*    io__,
//==                                    ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int io = *io__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_l_order(*l__, io);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         atom.symmetry_class().radial_function(ir, idxrf) = f__[ir];
//==     }
//== }
//==
//== void sirius_set_aw_radial_function_derivative(ftn_int*    ia__,
//==                                               ftn_int*    l__,
//==                                               ftn_int*    io__,
//==                                               ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int io = *io__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_l_order(*l__, io);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         atom.symmetry_class().radial_function_derivative(ir, idxrf) = f__[ir] * atom.type().radial_grid()[ir];
//==     }
//== }
//==
//== //void sirius_get_aw_deriv_radial_function(int32_t* ia__,
//== //                                         int32_t* l__,
//== //                                         int32_t* io__,
//== //                                         double* dfdr__)
//== //{
//== //    int ia = *ia__ - 1;
//== //    int io = *io__ - 1;
//== //    auto& atom = sim_ctx->unit_cell().atom(ia);
//== //    int idxrf = atom.type().indexr_by_l_order(*l__, io);
//== //    for (int ir = 0; ir < atom.num_mt_points(); ir++)
//== //    {
//== //        double rinv = atom.type().radial_grid().x_inv(ir);
//== //        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
//== //    }
//== //}
//==
//== void sirius_get_aw_surface_derivative(ftn_int*    ia__,
//==                                       ftn_int*    l__,
//==                                       ftn_int*    io__,
//==                                       ftn_int*    dm__,
//==                                       ftn_double* deriv__)
//== {
//==     *deriv__ = sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().aw_surface_dm(*l__, *io__ - 1, *dm__);
//== }
//==
//== void sirius_set_aw_surface_derivative(ftn_int*    ia__,
//==                                       ftn_int*    l__,
//==                                       ftn_int*    io__,
//==                                       ftn_int*    dm__,
//==                                       ftn_double* deriv__)
//== {
//==     sim_ctx->unit_cell().atom(*ia__ - 1).symmetry_class().set_aw_surface_deriv(*l__, *io__ - 1, *dm__, *deriv__);
//== }
//==
//== void sirius_get_lo_radial_function(ftn_int*    ia__,
//==                                    ftn_int*    idxlo__,
//==                                    ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxlo = *idxlo__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_idxlo(idxlo);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         f__[ir] = atom.symmetry_class().radial_function(ir, idxrf);
//==     }
//== }
//==
//== void sirius_set_lo_radial_function(ftn_int*    ia__,
//==                                    ftn_int*    idxlo__,
//==                                    ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxlo = *idxlo__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_idxlo(idxlo);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         atom.symmetry_class().radial_function(ir, idxrf) = f__[ir];
//==     }
//== }
//==
//== void sirius_set_lo_radial_function_derivative(ftn_int*    ia__,
//==                                               ftn_int*    idxlo__,
//==                                               ftn_double* f__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxlo = *idxlo__ - 1;
//==     auto& atom = sim_ctx->unit_cell().atom(ia);
//==     int idxrf = atom.type().indexr_by_idxlo(idxlo);
//==     for (int ir = 0; ir < atom.num_mt_points(); ir++) {
//==         atom.symmetry_class().radial_function_derivative(ir, idxrf) = f__[ir] * atom.type().radial_grid()[ir];
//==     }
//== }
//==
//== //void sirius_get_lo_deriv_radial_function(int32_t const* ia__,
//== //                                         int32_t const* idxlo__,
//== //                                         double* dfdr__)
//== //{
//== //    int ia = *ia__ - 1;
//== //    int idxlo = *idxlo__ - 1;
//== //    auto& atom = sim_ctx->unit_cell().atom(ia);
//== //    int idxrf = atom.type().indexr_by_idxlo(idxlo);
//== //    for (int ir = 0; ir < atom.num_mt_points(); ir++)
//== //    {
//== //        double rinv = atom.type().radial_grid().x_inv(ir);
//== //        dfdr__[ir] = atom.symmetry_class().r_deriv_radial_function(ir, idxrf) * rinv;
//== //    }
//== //}
//==
//== void sirius_get_aw_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* io1, int32_t* ilo2,
//==                                         double* oalo)
//== {
//==     int ia = *ia__ - 1;
//==
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
//==     int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;
//==
//==     *oalo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, *io1 - 1, order2);
//== }
//==
//== void sirius_set_aw_lo_o_radial_integral(int32_t* ia__,
//==                                         int32_t* l__,
//==                                         int32_t* io1__,
//==                                         int32_t* ilo2__,
//==                                         double* oalo__)
//== {
//==     int ia = *ia__ - 1;
//==
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
//==     int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;
//==
//==     //double d1 = std::abs(*oalo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, *io1__ - 1, order2));
//==     //double d2 = std::abs(*oalo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, order2, *io1__ - 1));
//==     //
//==     //if (d1 > 1e-6) {
//==     //    printf("ia: %i, oalo diff=%f\n", ia, d1);
//==     //}
//==     //if (d2 > 1e-6) {
//==     //    printf("ia: %i, oloa diff=%f\n", ia, d2);
//==     //}
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, *io1__ - 1, order2, *oalo__);
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order2, *io1__ - 1, *oalo__);
//== }
//==
//== void sirius_get_lo_lo_o_radial_integral(int32_t* ia__, int32_t* l, int32_t* ilo1, int32_t* ilo2,
//==                                         double* ololo)
//== {
//==     int ia = *ia__ - 1;
//==
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
//==     int order1 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf1).order;
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
//==     int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;
//==
//==     *ololo = sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l, order1, order2);
//== }
//==
//== void sirius_set_lo_lo_o_radial_integral(int32_t* ia__,
//==                                         int32_t* l__,
//==                                         int32_t* ilo1__,
//==                                         int32_t* ilo2__,
//==                                         double* ololo__)
//== {
//==     int ia = *ia__ - 1;
//==
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
//==     int order1 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf1).order;
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
//==     int order2 = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).order;
//==
//==     //double d1 = std::abs(*ololo__ - sim_ctx->unit_cell().atom(ia).symmetry_class().o_radial_integral(*l__, order1, order2));
//==     //
//==     //if (d1 > 1e-6) {
//==     //    printf("ia: %i, ololo diff=%f\n", ia, d1);
//==     //}
//==
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o_radial_integral(*l__, order1, order2, *ololo__);
//== }
//==
//== void sirius_get_aw_aw_h_radial_integral(int32_t* ia__, int32_t* l1, int32_t* io1, int32_t* l2,
//==                                         int32_t* io2, int32_t* lm3, double* haa)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l1, *io1 - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);
//==
//==     *haa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
//== }
//==
//== void sirius_set_aw_aw_h_radial_integral(int32_t* ia__,
//==                                         int32_t* l1__,
//==                                         int32_t* io1__,
//==                                         int32_t* l2__,
//==                                         int32_t* io2__,
//==                                         int32_t* lm3__,
//==                                         double* haa__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l1__, *io1__ - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *io2__ - 1);
//==
//==     //double d1 = std::abs(*haa__ - sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
//==     //
//==     //if (d1 > 1e-3) {
//==     //    printf("ia: %i, l1: %i, io1: %i, l2: %i, io2: %i, lm3: %i, haa diff=%f\n", ia, *l1__, *io1__, *l2__, *io2__, *lm3__, d1);
//==     //    printf("exciting value: %f, sirius value: %f\n", *haa__, sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
//==     //}
//==
//==     sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *haa__;
//== }
//==
//== void sirius_get_lo_aw_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* l2, int32_t* io2, int32_t* lm3,
//==                                         double* hloa)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2, *io2 - 1);
//==
//==     *hloa = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
//== }
//==
//== void sirius_set_lo_aw_h_radial_integral(int32_t* ia__,
//==                                         int32_t* ilo1__,
//==                                         int32_t* l2__,
//==                                         int32_t* io2__,
//==                                         int32_t* lm3__,
//==                                         double* hloa__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l2__, *io2__ - 1);
//==
//==     //double d1 = std::abs(*hloa__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
//==     //double d2 = std::abs(*hloa__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf2, idxrf1)[*lm3__ - 1]);
//==     //if (d1 > 1e-6) {
//==     //    printf("ia: %i, hloa diff=%f\n", ia, d1);
//==     //}
//==     //if (d2 > 1e-6) {
//==     //    printf("ia: %i, halo diff=%f\n", ia, d2);
//==     //}
//==
//==     sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *hloa__;
//==     sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf2, idxrf1)[*lm3__ - 1] = *hloa__;
//== }
//==
//==
//== void sirius_get_lo_lo_h_radial_integral(int32_t* ia__, int32_t* ilo1, int32_t* ilo2, int32_t* lm3,
//==                                         double* hlolo)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1 - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2 - 1);
//==
//==     *hlolo = sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3 - 1];
//== }
//==
//== void sirius_set_lo_lo_h_radial_integral(int32_t* ia__,
//==                                         int32_t* ilo1__,
//==                                         int32_t* ilo2__,
//==                                         int32_t* lm3__,
//==                                         double* hlolo__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
//==
//==     //double d1 = std::abs(*hlolo__ -  sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
//==     //if (d1 > 1e-6) {
//==     //    printf("ia: %i, lo1: %i, lo2: %i, lm3: %i, hlolo diff=%f\n", ia, *ilo1__, *ilo2__, *lm3__, d1);
//==     //    printf("exciting value: %f, sirius value: %f\n", *hlolo__, sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1]);
//==     //}
//==
//==     sim_ctx->unit_cell().atom(ia).h_radial_integrals(idxrf1, idxrf2)[*lm3__ - 1] = *hlolo__;
//== }
//==
//== void sirius_set_aw_aw_o1_radial_integral(ftn_int* ia__,
//==                                          ftn_int* l__,
//==                                          ftn_int* io1__,
//==                                          ftn_int* io2__,
//==                                          ftn_double* o1aa__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l__, *io1__ - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(*l__, *io2__ - 1);
//==
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1aa__);
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1aa__);
//== }
//==
//== void sirius_set_aw_lo_o1_radial_integral(ftn_int* ia__,
//==                                          ftn_int* io1__,
//==                                          ftn_int* ilo2__,
//==                                          ftn_double* o1alo__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
//==     int l = sim_ctx->unit_cell().atom(ia).type().indexr(idxrf2).l;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_l_order(l, *io1__ - 1);
//==
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1alo__);
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1alo__);
//== }
//==
//== void sirius_set_lo_lo_o1_radial_integral(ftn_int* ia__,
//==                                          ftn_int* ilo1__,
//==                                          ftn_int* ilo2__,
//==                                          ftn_double* o1lolo__)
//== {
//==     int ia = *ia__ - 1;
//==     int idxrf1 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo1__ - 1);
//==     int idxrf2 = sim_ctx->unit_cell().atom(ia).type().indexr_by_idxlo(*ilo2__ - 1);
//==
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf1, idxrf2, *o1lolo__);
//==     sim_ctx->unit_cell().atom(ia).symmetry_class().set_o1_radial_integral(idxrf2, idxrf1, *o1lolo__);
//== }
//==
//== //void sirius_generate_potential_pw_coefs()
//== //{
//== //    STOP();
//== //    //potential->generate_pw_coefs();
//== //}
//== //
//== //void sirius_generate_density_pw_coefs()
//== //{
//== //    STOP();
//== //    //density->generate_pw_coefs();
//== //}
//==
//== /// Get first-variational eigen-vectors
//== /** Assume that the Fortran side holds the whole array */
//== void sirius_get_fv_eigen_vectors(int32_t* kset_id__, int32_t* ik__, double_complex* fv_evec__, int32_t* ld__,
//==                                  int32_t* num_fv_evec__)
//== {
//==     mdarray<double_complex, 2> fv_evec(fv_evec__, *ld__, *num_fv_evec__);
//==     (*kset_list[*kset_id__])[*ik__ - 1]->get_fv_eigen_vectors(fv_evec);
//== }
//==
//== /// Get second-variational eigen-vectors
//== /** Assume that the Fortran side holds the whole array */
//== void sirius_get_sv_eigen_vectors(int32_t* kset_id, int32_t* ik, double_complex* sv_evec__, int32_t* size)
//== {
//==     mdarray<double_complex, 2> sv_evec(sv_evec__, *size, *size);
//==     (*kset_list[*kset_id])[*ik - 1]->get_sv_eigen_vectors(sv_evec);
//== }
//==
//== void sirius_get_num_fv_states(int32_t* num_fv_states__)
//== {
//==     *num_fv_states__ = sim_ctx->num_fv_states();
//== }
//==
//== //void sirius_get_mpi_comm(int32_t* directions__, int32_t* fcomm__)
//== //{
//== //    *fcomm__ = MPI_Comm_c2f(sim_ctx->mpi_grid().communicator(*directions__).mpi_comm());
//== //}
//==
//== void sirius_get_fft_comm(int32_t* fcomm__)
//== {
//==     *fcomm__ = MPI_Comm_c2f(sim_ctx->fft().comm().mpi_comm());
//== }
//==
//== void sirius_get_kpoint_inner_comm(int32_t* fcomm__)
//== {
//==     *fcomm__ = MPI_Comm_c2f(sim_ctx->comm_band().mpi_comm());
//== }
//==
//== void sirius_get_all_kpoints_comm(int32_t* fcomm__)
//== {
//==     *fcomm__ = MPI_Comm_c2f(sim_ctx->comm_k().mpi_comm());
//== }
//==
//== void sirius_forces(double* forces__)
//== {
//==     //mdarray<double, 2> forces(forces__, 3, sim_ctx->unit_cell().num_atoms());
//==     //dft_ground_state->forces(forces);
//== }
//==
//==
//== void sirius_core_leakage(double* core_leakage)
//== {
//==     *core_leakage = dft_ground_state->density().core_leakage();
//== }
//==
//== void sirius_ground_state_print_info()
//== {
//==     dft_ground_state->print_info();
//== }
//==
//== void sirius_create_storage_file()
//== {
//==     sim_ctx->create_storage_file();
//== }
//==
//==
//== //void sirius_density_mixer_initialize(void)
//== //{
//== //    dft_ground_state->density().mixer_init(sim_ctx->mixer_input());
//== //}
//== //
//== //void sirius_mix_density(double* rms)
//== //{
//== //    *rms = dft_ground_state->density().mix();
//== //    dft_ground_state->density().fft_transform(1);
//== //    sim_ctx->comm().bcast(rms, 1, 0);
//== //}
//== //
//==
//==
//== void sirius_symmetrize_density()
//== {
//==     dft_ground_state->density().symmetrize();
//== }
//==
//== void sirius_get_gvec_index(int32_t* gvec__, int32_t* ig__)
//== {
//==     vector3d<int> gv(gvec__[0], gvec__[1], gvec__[2]);
//==     *ig__ = sim_ctx->gvec().index_by_gvec(gv) + 1;
//== }
//==
//== void sirius_use_internal_mixer(int32_t* use_internal_mixer__)
//== {
//==     *use_internal_mixer__ = (sim_ctx->mixer_input().exist_) ? 1 : 0;
//== }
//==
//==
//== void sirius_set_iterative_solver_type(ftn_char type__)
//== {
//==     sim_ctx->set_iterative_solver_type(std::string(type__));
//== }
//==
//== void sirius_get_density_dr2(double* dr2__)
//== {
//==     *dr2__ = dft_ground_state->density().dr2();
//== }
//==
//== void sirius_real_gaunt_coeff_(int32_t* lm1__, int32_t* lm2__, int32_t* lm3__, double* coeff__)
//== {
//==     std::vector<int> idxlm(100);
//==     std::vector<int> phase(100, 1);
//==     int lm = 0;
//==     for (int l = 0; l < 10; l++)
//==     {
//==         idxlm[lm++] = Utils::lm_by_l_m(l, 0);
//==         for (int m = 1; m <= l; m++)
//==         {
//==             idxlm[lm++] = Utils::lm_by_l_m(l, m);
//==             idxlm[lm] = Utils::lm_by_l_m(l, -m);
//==             if (m % 2 == 0) phase[lm] = -1;
//==             lm++;
//==         }
//==     }
//==
//==     int l1(0), m1(0), l2(0), m2(0), l3(0), m3(0);
//==     int s = 1;
//==
//==     for (int l = 0; l < 10; l++)
//==     {
//==         for (int m = -l; m <= l; m++)
//==         {
//==             int lm = Utils::lm_by_l_m(l, m);
//==             if (lm == idxlm[*lm1__ - 1])
//==             {
//==                 l1 = l;
//==                 m1 = m;
//==                 s *= phase[*lm1__ - 1];
//==             }
//==             if (lm == idxlm[*lm2__ - 1])
//==             {
//==                 l2 = l;
//==                 m2 = m;
//==                 s *= phase[*lm2__ - 1];
//==             }
//==             if (lm == idxlm[*lm3__ - 1])
//==             {
//==                 l3 = l;
//==                 m3 = m;
//==                 s *= phase[*lm3__ - 1];
//==             }
//==         }
//==     }
//==     double d = 0;
//==     for (int k1 = -l1; k1 <= l1; k1++)
//==     {
//==         for (int k2 = -l2; k2 <= l2; k2++)
//==         {
//==             for (int k3 = -l3; k3 <= l3; k3++)
//==             {
//==                 d += real(conj(sirius::SHT::ylm_dot_rlm(l1, k1, m1)) *
//==                           sirius::SHT::ylm_dot_rlm(l2, k2, m2) *
//==                           sirius::SHT::ylm_dot_rlm(l3, k3, m3)) * sirius::SHT::gaunt_ylm(l1, l2, l3, k1, k2, k3);
//==             }
//==         }
//==     }
//==     //double d = sirius::SHT::gaunt<double>(l1, l2, l3, m1, m2, m3);
//==
//==     *coeff__ = d * s;
//== }
//==
//== void sirius_ylmr2_(int32_t* lmmax__, int32_t* nr__, double* vr__, double* rlm__)
//== {
//==     mdarray<double, 2> rlm(rlm__, *nr__, *lmmax__);
//==     mdarray<double, 2> vr(vr__, 3, *nr__);
//==
//==     int lmax = Utils::lmax_by_lmmax(*lmmax__);
//==
//==     std::vector<int> idxlm(*lmmax__);
//==     std::vector<int> phase(*lmmax__, 1);
//==     int lm = 0;
//==     for (int l = 0; l <= lmax; l++)
//==     {
//==         idxlm[lm++] = Utils::lm_by_l_m(l, 0);
//==         for (int m = 1; m <= l; m++)
//==         {
//==             idxlm[lm++] = Utils::lm_by_l_m(l, m);
//==             idxlm[lm] = Utils::lm_by_l_m(l, -m);
//==             if (m % 2 == 0) phase[lm] = -1;
//==             lm++;
//==         }
//==     }
//==
//==     std::vector<double> rlm_tmp(*lmmax__);
//==     for (int i = 0; i < *nr__; i++)
//==     {
//==         auto vs = sirius::SHT::spherical_coordinates(vector3d<double>(vr(0, i), vr(1, i), vr(2, i)));
//==         sirius::SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm_tmp[0]);
//==         for (int lm = 0; lm < *lmmax__; lm++) rlm(i, lm) = rlm_tmp[idxlm[lm]] * phase[lm];
//==     }
//== }
//==
//== //void sirius_get_vloc_(int32_t* size__, double* vloc__)
//== //{
//== //    TERMINATE("fix this");
//== //    //if (!sim_ctx) return;
//== //
//== //    //auto fft_coarse = sim_ctx->fft_coarse();
//== //    //if (*size__ != fft_coarse->size())
//== //    //{
//== //    //    TERMINATE("wrong size of coarse FFT mesh");
//== //    //}
//== //
//== //    ///* map effective potential to a corase grid */
//== //    //std::vector<double> veff_it_coarse(fft_coarse->size());
//== //    //std::vector<double_complex> veff_pw_coarse(fft_coarse->num_gvec());
//== //
//== //    ///* take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi */
//== //    //for (int igc = 0; igc < fft_coarse->num_gvec(); igc++)
//== //    //{
//== //    //    int ig = sim_ctx->fft().gvec_index(fft_coarse->gvec(igc));
//== //    //    veff_pw_coarse[igc] = dft_ground_state->potential().effective_potential()->f_pw(ig);
//== //    //}
//== //    //fft_coarse->input(fft_coarse->num_gvec(), fft_coarse->index_map(), &veff_pw_coarse[0]);
//== //    //fft_coarse->transform(1);
//== //    //fft_coarse->output(vloc__);
//== //    //for (int i = 0; i < fft_coarse->size(); i++) vloc__[i] *= 2; // convert to Ry
//== //}
//==
//==
//== void sirius_get_fv_states_(int32_t* kset_id__, int32_t* ik__, int32_t* nfv__, int32_t* ngk__, int32_t* gvec_of_k__,
//==                            double_complex* fv_states__, int32_t* ld__)
//== {
//==     TERMINATE("fix this");
//==     //auto kset = kset_list[*kset_id__];
//==     //auto kp = (*kset)[*ik__ - 1];
//==
//==     //if (*ngk__ != kp->num_gkvec())
//==     //{
//==     //    std::stringstream s;
//==     //    s << "wrong number of G+k vectors" << std::endl
//==     //      << "ik = " << *ik__ - 1 << std::endl
//==     //      << "ngk = " << *ngk__ << std::endl
//==     //      << "kp->num_gkvec() = " << kp->num_gkvec();
//==     //    TERMINATE(s);
//==     //}
//==     //if (*nfv__ != sim_ctx->num_fv_states())
//==     //{
//==     //    TERMINATE("wrong number of first-variational states");
//==     //}
//==     //
//==     //mdarray<int, 2> gvec_of_k(gvec_of_k__, 3, *ngk__);
//==     //std::vector<int> igk_map(*ngk__);
//==     //for (int igk = 0; igk < kp->num_gkvec(); igk++)
//==     //{
//==     //    bool found = false;
//==     //    for (int i = 0; i < kp->num_gkvec(); i++)
//==     //    {
//==     //        int ig = kp->gvec_index(i);
//==     //        /* G-vector of sirius ordering */
//==     //        auto vg = sim_ctx->fft().gvec(ig);
//==     //        if (gvec_of_k(0, igk) == vg[0] &&
//==     //            gvec_of_k(1, igk) == vg[1] &&
//==     //            gvec_of_k(2, igk) == vg[2])
//==     //        {
//==     //            igk_map[igk] = i;
//==     //            found = true;
//==     //        }
//==     //    }
//==     //    if (!found)
//==     //    {
//==     //        TERMINATE("G-vector is not found");
//==     //    }
//==     //}
//==
//==     //mdarray<double_complex, 2> fv_states(fv_states__, *ld__, *nfv__);
//==
//==     //for (int i = 0; i < sim_ctx->num_fv_states(); i++)
//==     //{
//==     //    for (int igk = 0; igk < kp->num_gkvec(); igk++)
//==     //    {
//==     //        fv_states(igk, i) = kp->fv_states()(igk_map[igk], i);
//==     //    }
//==     //}
//== }
//==
//==
//==
//== void sirius_get_paw_total_energy(double* tot_en__)
//== {
//==     *tot_en__ = dft_ground_state->potential().PAW_total_energy();
//== }
//==
//== void sirius_get_paw_one_elec_energy(double* one_elec_en__)
//== {
//==     *one_elec_en__ = dft_ground_state->potential().PAW_one_elec_energy();
//== }
//==
//== void sirius_reduce_coordinates(ftn_double* coord__,
//==                                ftn_double* reduced_coord__,
//==                                ftn_int* T__)
//== {
//==     vector3d<double> coord(coord__[0], coord__[1], coord__[2]);
//==     auto result = reduce_coordinates(coord);
//==     for (int x: {0, 1, 2}) {
//==         reduced_coord__[x] = result.first[x];
//==         T__[x] = result.second[x];
//==     }
//== }
//==
//== void sirius_fderiv(ftn_int* m__,
//==                    ftn_int* np__,
//==                    ftn_double* x__,
//==                    ftn_double* f__,
//==                    ftn_double* g__)
//== {
//==     int np = *np__;
//==     sirius::Radial_grid_ext<double> rgrid(np, x__);
//==     sirius::Spline<double> s(rgrid);
//==     for (int i = 0; i < np; i++) {
//==         s(i) = f__[i];
//==     }
//==     s.interpolate();
//==     switch (*m__) {
//==         case -1: {
//==             std::vector<double> g(np);
//==             s.integrate(g, 0);
//==             for (int i = 0; i < np; i++) {
//==                 g__[i] = g[i];
//==             }
//==             return;
//==         }
//==         default: {
//==              TERMINATE_NOT_IMPLEMENTED;
//==         }
//==     }
//== }
//==
//==
//== void sirius_get_beta_projectors(ftn_int*            kset_id__,
//==                                 ftn_int*            ik__,
//==                                 ftn_int*            npw__,
//==                                 ftn_int*            gvec_k__,
//==                                 ftn_double_complex* vkb__,
//==                                 ftn_int*            ld__,
//==                                 ftn_int*            nkb__)
//== {
//==     PROFILE("sirius_api::sirius_get_beta_projectors");
//==
//==     if (*nkb__ != sim_ctx->unit_cell().mt_lo_basis_size()) {
//==         TERMINATE("wrong number of beta-projectors");
//==     }
//==
//==     auto kset = kset_list[*kset_id__];
//==     auto kp = (*kset)[*ik__ - 1];
//==
//==     mdarray<int, 2> gvec_k(gvec_k__, 3, *npw__);
//==     mdarray<double_complex, 2> vkb(vkb__, *ld__, *nkb__);
//==     vkb.zero();
//==
//==     auto& gkvec = kp->gkvec();
//==
//==     /* list of sirius G-vector indices which fall into cutoff |G+k| < Gmax */
//==     std::vector<int> idxg;
//==     /* mapping  between QE and sirius indices */
//==     std::vector<int> idxg_map(*npw__, -1);
//==     /* loop over all input G-vectors */
//==     for (int i = 0; i < *npw__; i++) {
//==         /* take input G-vector + k-vector */
//==         auto gvc = sim_ctx->unit_cell().reciprocal_lattice_vectors() * (vector3d<double>(gvec_k(0, i), gvec_k(1, i), gvec_k(2, i)) + kp->vk());
//==         /* skip it if its length is larger than the cutoff */
//==         if (gvc.length() > sim_ctx->gk_cutoff()) {
//==             continue;
//==         }
//==         /* get index of G-vector */
//==         int ig = gkvec.index_by_gvec({gvec_k(0, i), gvec_k(1, i), gvec_k(2, i)});
//==         if (ig == -1) {
//==             TERMINATE("index of G-vector is not found");
//==         }
//==         idxg_map[i] = static_cast<int>(idxg.size());
//==         idxg.push_back(ig);
//==     }
//==
//==     sirius::Beta_projectors bp(*sim_ctx, gkvec, idxg);
//==     bp.prepare();
//==     bp.generate(0);
//==     auto& beta_a = bp.pw_coeffs_a();
//==
//==     for (int ia = 0; ia < sim_ctx->unit_cell().num_atoms(); ia++) {
//==         auto& atom = sim_ctx->unit_cell().atom(ia);
//==         int nbf = atom.mt_basis_size();
//==
//==         auto qe_order = atomic_orbital_index_map_QE(atom.type());
//==
//==         for (int xi = 0; xi < nbf; xi++) {
//==             for (int i = 0; i < *npw__; i++) {
//==                 if (idxg_map[i] != -1) {
//==                     vkb(i, atom.offset_lo() + qe_order[xi]) = beta_a(idxg_map[i], atom.offset_lo() + xi) * static_cast<double>(phase_Rlm_QE(atom.type(), xi));
//==                 } else {
//==                     vkb(i, atom.offset_lo() + qe_order[xi]) = 0;
//==                 }
//==             }
//==         }
//==     }
//== }
//==
//== void sirius_get_beta_projectors_by_kp(ftn_int*            kset_id__,
//==                                       ftn_double*         vk__,
//==                                       ftn_int*            npw__,
//==                                       ftn_int*            gvec_k__,
//==                                       ftn_double_complex* vkb__,
//==                                       ftn_int*            ld__,
//==                                       ftn_int*            nkb__)
//== {
//==     PROFILE("sirius_api::sirius_get_beta_projectors_by_kp");
//==
//==     vector3d<double> vk(vk__[0], vk__[1], vk__[2]);
//==
//==     auto kset = kset_list[*kset_id__];
//==     for (int ikloc = 0; ikloc < kset->spl_num_kpoints().local_size(); ikloc++) {
//==         int ik = kset->spl_num_kpoints(ikloc);
//==         auto kp = (*kset)[ik];
//==         if ((kp->vk() - vk).length() < 1e-12) {
//==             int k = ik + 1;
//==             sirius_get_beta_projectors(kset_id__, &k, npw__, gvec_k__, vkb__, ld__, nkb__);
//==             return;
//==         }
//==     }
//==     std::stringstream s;
//==     s << "k-point " << vk << " is not found" << std::endl
//==       << "mpi rank: " << kset->comm().rank() << std::endl
//==       << "list of local k-points : " << std::endl;
//==     for (int ikloc = 0; ikloc < kset->spl_num_kpoints().local_size(); ikloc++) {
//==         int ik = kset->spl_num_kpoints(ikloc);
//==         auto kp = (*kset)[ik];
//==         s << kp->vk() << std::endl;
//==     }
//==     TERMINATE(s);
//== }
//==
//==
//== void sirius_set_processing_unit(ftn_char pu__)
//== {
//==     sim_ctx->set_processing_unit(pu__);
//== }
//==
//== void sirius_calculate_hubbard_potential()
//== {
//==     dft_ground_state->hamiltonian().U().calculate_hubbard_potential_and_energy();
//== }
//==
//== void sirius_spline_(ftn_int* n__, ftn_double* x__, ftn_double* f__, ftn_double* cf__)
//== {
//==     int np = *n__;
//==
//==     sirius::Radial_grid_ext<double> rgrid(np, x__);
//==     sirius::Spline<double> s(rgrid, std::vector<double>(f__, f__ + np));
//==
//==     mdarray<double, 2> cf(cf__, 3, np);
//==
//==     for (int i = 0; i < np - 1; i++) {
//==         auto c = s.coeffs(i);
//==         cf(0, i) = c[1];
//==         cf(1, i) = c[2];
//==         cf(2, i) = c[3];
//==     }
//== }
//==
} // extern "C"

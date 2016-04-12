include ../make.inc

.SUFFIXES: .cpp .cu .f90

SRC =  LebedevLaikov.cpp linalg.cpp utils.cpp radial_grid.cpp \
       radial_solver.cpp sht.cpp step_function.cpp band.cpp potential.cpp k_set.cpp symmetry.cpp \
       xc_functional.cpp spheric_function.cpp force.cpp dft_ground_state.cpp real_space_prj.cpp fft3d.cpp \
       simulation_context.cpp wave_functions.cpp beta_projectors.cpp non_local_operator.cpp \
       Atom_type/add_aw_descriptor.cpp \
       Atom_type/add_lo_descriptor.cpp \
       Atom_type/atom_type.cpp \
       Atom_type/init.cpp \
       Atom_type/init_aw_descriptors.cpp \
       Atom_type/init_free_atom.cpp \
       Atom_type/print_info.cpp \
       Atom_type/read_input.cpp \
       Atom_type/set_radial_grid.cpp \
       Atom/atom.cpp \
       Atom/init.cpp \
       Atom/generate_radial_integrals.cpp \
       Unit_cell/atom_symmetry_class.cpp \
       Unit_cell/unit_cell.cpp \
       Unit_cell/make_periodic_function.cpp \
       Density/density.cpp \
       Density/initial_density.cpp \
       Density/add_k_point_contribution.cpp \
       Density/add_k_point_contribution_rg.cpp \
       Density/generate_rho_radial_integrals.cpp \
       Density/reduce_density_matrix.cpp \
       Density/generate.cpp \
       Density/generate_valence.cpp \
       Density/generate_valence_density_mt.cpp \
       Density/generate_valence_density_it.cpp \
       Density/generate_pseudo_core_charge_density.cpp \
       Density/augment.cpp \
       Density/set_magnetization_ptr.cpp \
       Density/generate_core_charge_density.cpp \
       Density/check_density_continuity_at_mt.cpp \
       Band/set_fv_h_o.cpp \
       Band/set_h_o.cpp \
       Band/add_nl_h_o_pw.cpp \
       Band/add_nl_h_o_rs.cpp \
       Band/apply_h_o.cpp \
       Band/apply_h_serial.cpp \
       Band/apply_h_o_fast_parallel_rs.cpp \
       Band/get_h_o_diag.cpp \
       Band/diag_pseudo_potential.cpp \
       Band/diag_pseudo_potential_exact.cpp \
       Band/diag_pseudo_potential_davidson.cpp \
       Band/diag_pseudo_potential_rmm_diis.cpp \
       Band/diag_h_o.cpp \
       Band/apply_magnetic_field.cpp \
       Band/residuals_aux.cpp \
       Band/solve_sv.cpp \
       Band/residuals.cpp \
       Band/diag_fv_full_potential.cpp \
       Band/orthogonalize.cpp \
       Band/initialize_subspace.cpp \
       Potential/init.cpp \
       Potential/poisson.cpp \
       Potential/xc.cpp \
       Potential/generate_d_operator_matrix.cpp \
       Potential/generate_local_potential.cpp \
       Potential/generate_effective_potential.cpp \
       K_point/distribute_basis_index.cpp \
       K_point/build_gklo_basis_descriptors.cpp \
       K_point/generate_fv_states.cpp \
       K_point/generate_spinor_wave_functions.cpp \
       K_point/generate_gkvec.cpp \
       K_point/initialize.cpp \
       K_point/k_point.cpp \
       K_point/test_fv_states.cpp \
       K_point/num_occupied_bands.cpp 

OBJ = $(SRC:.cpp=.o) 





ifneq (,$(findstring D__GPU,$(CXX_OPT)))
	OBJ := $(OBJ) \
	Kernels/cuda_interface.o \
	Kernels/cuda_uspp_kernels.o \
	Kernels/scale_matrix.o \
	Kernels/spline.o \
	Kernels/create_beta_gk.o \
	Kernels/mul_veff_with_phase_factors.o \
	Kernels/density_rg.o \
	Kernels/pack_unpack_z_columns.o \
	Kernels/residuals_aux.o \
	Kernels/generate_dm_pw.o \
	Kernels/sum_q_pw_dm_pw.o \
	Kernels/generate_phase_factors.o
endif

ifneq (,$(findstring D__MAGMA,$(CXX_OPT)))
	OBJ := $(OBJ) Kernels/magma_interface.o
endif

ifneq (,$(findstring D__PLASMA,$(CXX_OPT)))
	OBJ := $(OBJ) plasma_interface.o
endif


OBJ_DBG = $(OBJ:%.o=%.o)
	
#.cpp.o:
#	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ $(NVCC_OPT) -c -o $(*D)/$(*F).o $<

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c $<

all: log $(OBJ) sirius_api.o sirius.o
	ar -r libsirius.a $(OBJ)
	ar -r libsirius_f.a $(OBJ) sirius_api.o sirius.o



%.o: %.cpp
	$(CXX) $(CXX_OPT_DBG) -c -o $(*D)/$(*F).o $<
	
debug: $(OBJ_DBG)
	ar -r libsirius.dbg.a $(OBJ_DBG)


	
# TODO: platform-independent python script
log:
	@echo "#ifndef __VERSION_H__" > version.h
	@echo "#define __VERSION_H__" >> version.h
	@echo "const char* const git_hash = \"$(shell git rev-parse HEAD)\";" >> version.h 
	@echo "const char* const build_date = \"$(shell date)\";" >> version.h 
	@echo "#endif" >> version.h

clean:
	rm -f $(OBJ) *.o *.a *.mod

depend: $(SRC) sirius_api.cpp
	rm -f ./.depend
	$(CXX) $(CXX_OPT) -MM $^ >> ./.depend

showlibs:
	@echo
	@echo List of libraries for linking with the Fortran code:
	@echo
	@echo $(shell pwd)/libsirius_f.a $(LIBS) 

ifneq ($(wildcard .depend),)
include .depend
endif

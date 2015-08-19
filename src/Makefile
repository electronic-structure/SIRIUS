include ../make.inc

.SUFFIXES: .cpp .cu .f90

SRC =  LebedevLaikov.cpp linalg.cpp utils.cpp error_handling.cpp timer.cpp platform.cpp radial_grid.cpp \
       radial_solver.cpp sht.cpp \
       reciprocal_lattice.cpp step_function.cpp band.cpp potential.cpp \
       k_set.cpp density.cpp symmetry.cpp xc_functional.cpp spheric_function.cpp force.cpp dft_ground_state.cpp \
       real_space_prj.cpp \
       Unit_cell/atom.cpp \
       Unit_cell/atom_type.cpp \
       Unit_cell/atom_symmetry_class.cpp \
       Unit_cell/unit_cell.cpp \
       Density/density.cpp \
       Density/initial_density.cpp \
       Density/add_k_point_contribution.cpp \
       Density/add_k_point_contribution_it.cpp \
       Density/generate_rho_radial_integrals.cpp \
       Density/reduce_density_matrix.cpp \
       Density/add_q_contribution_to_valence_density.cpp \
       Density/generate_valence.cpp \
       Density/generate_valence_density_mt.cpp \
       Density/generate_valence_density_it.cpp \
       Density/generate_pseudo_core_charge_density.cpp \
       Band/apply_h_local_serial.cpp \
       Band/apply_h_local_parallel.cpp \
       Band/set_fv_h_o_serial.cpp \
       Band/set_fv_h_o_parallel.cpp \
       Band/add_nl_h_o_pw.cpp \
       Band/add_nl_h_o_rs.cpp \
       Band/add_non_local_contribution_serial.cpp \
       Band/add_non_local_contribution_parallel.cpp \
       Band/apply_h_o_serial.cpp \
       Band/apply_h_serial.cpp \
       Band/apply_h_o_parallel.cpp \
       Band/apply_h_o_fast_parallel_rs.cpp \
       Band/diag_fv_pseudo_potential_exact_serial.cpp \
       Band/diag_fv_pseudo_potential_rmm_diis_serial.cpp \
       Band/diag_fv_pseudo_potential_chebyshev_serial.cpp \
       Band/diag_fv_pseudo_potential_chebyshev_parallel.cpp \
       Band/diag_fv_pseudo_potential_davidson_serial.cpp \
       Band/diag_fv_pseudo_potential_davidson_parallel.cpp \
       Band/apply_magnetic_field.cpp \
       Band/residuals_serial.cpp \
       Band/residuals_parallel.cpp \
       Band/solve_sv.cpp \
       Band/diag_fv_full_potential.cpp \
       Potential/init.cpp \
       Potential/poisson.cpp \
       Potential/xc.cpp \
       Potential/generate_d_operator_matrix.cpp \
       Potential/generate_local_potential.cpp \
       K_point/add_non_local_contribution.cpp \
       K_point/generate_beta_gk.cpp \
       K_point/generate_beta_gk_t.cpp \
       K_point/generate_beta_phi.cpp \
       K_point/distribute_basis_index.cpp \
       K_point/build_gklo_basis_descriptors.cpp \
       K_point/generate_fv_states.cpp \
       K_point/generate_spinor_wave_functions.cpp \
       K_point/generate_gkvec.cpp \
       K_point/init_gkvec.cpp \
       K_point/initialize.cpp \
       K_point/k_point.cpp \
       K_point/test_fv_states.cpp \
       K_point/get_occupied_bands_list.cpp

OBJ = $(SRC:.cpp=.o) 

ifneq (,$(findstring D__GPU,$(CXX_OPT)))
	OBJ := $(OBJ) \
	Kernels/cuda_interface.o \
	Kernels/cuda_uspp_kernels.o \
	Kernels/compute_residuals.o \
	Kernels/apply_preconditioner.o \
	Kernels/normalize_residuals.o \
	Kernels/scale_matrix.o \
	Kernels/spline.o \
	Kernels/create_beta_gk.o \
	Kernels/mul_veff_with_phase_factors.o
endif

ifneq (,$(findstring D__MAGMA,$(CXX_OPT)))
	OBJ := $(OBJ) Kernels/magma_interface.o
endif

ifneq (,$(findstring D__PLASMA,$(CXX_OPT)))
	OBJ := $(OBJ) plasma_interface.o
endif

	
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ $(NVCC_OPT) -c -o $(*D)/$(*F).o $<

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c $<

all: log $(OBJ) sirius_api.o sirius.o
	ar -r libsirius.a $(OBJ)
	ar -r libsirius_f.a $(OBJ) sirius_api.o sirius.o

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

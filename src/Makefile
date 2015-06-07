include ../make.inc

.SUFFIXES: .cpp .cu .f90

SRC =  LebedevLaikov.cpp linalg.cpp utils.cpp error_handling.cpp timer.cpp platform.cpp radial_grid.cpp \
       radial_solver.cpp sht.cpp atom_type.cpp atom_symmetry_class.cpp atom.cpp unit_cell.cpp \
       reciprocal_lattice.cpp step_function.cpp k_point.cpp band.cpp potential.cpp \
       k_set.cpp density.cpp symmetry.cpp xc_functional.cpp spheric_function.cpp force.cpp dft_ground_state.cpp \
       band_uspp_gpu_parallel.cpp band_pseudo_pot.cpp band_pseudo_pot_itsol.cpp real_space_prj.cpp \
       Density/density.cpp \
       Density/initial_density.cpp \
       Density/add_k_point_contribution.cpp \
       Density/add_k_point_contribution_it.cpp \
       Density/generate_rho_radial_integrals.cpp \
       Density/reduce_density_matrix.cpp \
       Density/get_occupied_bands_list.cpp \
       Density/add_q_contribution_to_valence_density.cpp \
       Density/generate_valence_density_mt.cpp \
       Density/generate_valence_density_it.cpp \
       Band/apply_h_local_slice.cpp \
       Band/set_fv_h_o_serial.cpp \
       Band/add_nl_h_o_pw.cpp \
       Band/add_nl_h_o_rs.cpp \
       Band/apply_h_o_serial.cpp \
       Band/diag_fv_pseudo_potential_rmm_diis_serial.cpp \
       Band/diag_fv_pseudo_potential_chebyshev_serial.cpp \
       Band/diag_fv_pseudo_potential_davidson_serial.cpp \
       Band/diag_fv_pseudo_potential_davidson_parallel.cpp \
       Band/diag_fv_pseudo_potential_davidson_fast_parallel.cpp \
       Band/apply_h_o_fast_parallel.cpp \
       Band/apply_h_o_fast_parallel_rs.cpp \
       Band/apply_magnetic_field.cpp \
       Band/residuals_serial.cpp \
       Potential/poisson.cpp \
       Potential/xc.cpp \
       K_point/add_non_local_contribution.cpp \
       K_point/collect_all_bands.cpp \
       K_point/collect_all_gkvec.cpp \
       K_point/generate_beta_gk.cpp \
       K_point/generate_beta_phi.cpp
       
OBJ = $(SRC:.cpp=.o) 
      
ifneq (,$(findstring D__GPU,$(CXX_OPT)))
	OBJ := $(OBJ) \
	cuda_interface.o \
	cuda_uspp_kernels.o \
	cuda_spline_kernels.o \
	Kernels/compute_residuals.o \
	Kernels/apply_preconditioner.o \
	Kernels/normalize_residuals.o \
	Kernels/scale_matrix.o
endif

ifneq (,$(findstring D__MAGMA,$(CXX_OPT)))
	OBJ := $(OBJ) magma_interface.o
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

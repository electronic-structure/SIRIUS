include ../make.inc

.SUFFIXES: .cpp .cu .f90

SRC =  LebedevLaikov.cpp \
       potential.cpp symmetry.cpp \
       spheric_function.cpp force.cpp dft_ground_state.cpp real_space_prj.cpp \
       beta_projectors.cpp \
       atom_type.cpp atom.cpp atom_symmetry_class.cpp unit_cell.cpp \
       Potential/potential.cpp \
       Potential/poisson.cpp \
       Potential/xc.cpp \
       Potential/generate_local_potential.cpp \
       Potential/generate_effective_potential.cpp \
       Potential/paw_potential.cpp

OBJ = $(SRC:.cpp=.o)





ifneq (,$(findstring D__GPU,$(CXX_OPT)))
	OBJ := $(OBJ) \
	Kernels/cuda_interface.o \
	Kernels/cufft_interface.o \
	Kernels/cublas_interface.o \
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
	Kernels/generate_phase_factors.o \
    Kernels/add_pw_ekin.o
endif

ifneq (,$(findstring D__MAGMA,$(CXX_OPT)))
	OBJ := $(OBJ) Kernels/magma_interface.o
endif

ifneq (,$(findstring D__PLASMA,$(CXX_OPT)))
	OBJ := $(OBJ) plasma_interface.o
endif


OBJ_DBG = $(OBJ:%.o=%.o)

.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ $(NVCC_OPT) -c -o $(*D)/$(*F).o $<

.f90.o:
	$(MPI_FC) $(MPI_FC_OPT) -c $<

all: log $(OBJ) sirius_api.o sirius.o
	ar -r libsirius.a $(OBJ)
	ar -r libsirius_f.a $(OBJ) sirius_api.o sirius.o



#%.o: %.cpp
#	$(CXX) $(CXX_OPT_DBG) -c -o $(*D)/$(*F).o $<

debug: CXX_OPT=$(CXX_OPT_DBG)
debug: $(OBJ_DBG)
	ar -r libsirius.dbg.a $(OBJ_DBG)



# TODO: platform-independent python script
log:
	@echo "#ifndef __VERSION_H__" > version.h
	@echo "#define __VERSION_H__" >> version.h
	@echo "const char* const git_hash = \"$(shell git rev-parse HEAD)\";" >> version.h
	@echo "const char* const build_date = \"$(shell date +"%a, %e %b %Y %H:%M:%S")\";" >> version.h
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

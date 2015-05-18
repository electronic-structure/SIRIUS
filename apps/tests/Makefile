include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: hdf5 allgather mt_function mt_kinetic spline splindex hydrogen read_atom fft fft_t test_mdarray test_xc

hdf5: hdf5.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
allgather: allgather.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
mt_function: mt_function.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
mt_kinetic: mt_kinetic.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
spline: spline.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_zgemm: test_zgemm.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
splindex: splindex.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
hydrogen: hydrogen.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
read_atom:
	$(CXX) $(CXX_OPT) $(INCLUDE) read_atom.cpp     $(LIB_SIRIUS) $(LIBS) -o read_atom
	
fft: fft.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) fft.cpp           $(LIB_SIRIUS) $(LIBS) -o fft
	
fft_t: fft_t.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
	
test_mdarray:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_mdarray.cpp  $(LIB_SIRIUS) $(LIBS) -o test_mdarray

test_pstdout:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_pstdout.cpp  $(LIB_SIRIUS) $(LIBS) -o test_pstdout

test_init:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_init.cpp  $(LIB_SIRIUS) $(LIBS) -o test_init

test_init_global:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_init_global.cpp  $(LIB_SIRIUS) $(LIBS) -o test_init_global

test_blacs:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_blacs.cpp  $(LIB_SIRIUS) $(LIBS) -o test_blacs

test_iallgather:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_iallgather.cpp  $(LIB_SIRIUS) $(LIBS) -o test_iallgather

test_diag:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_diag.cpp  $(LIB_SIRIUS) $(LIBS) -o test_diag

test_rank2:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_rank2.cpp  $(LIB_SIRIUS) $(LIBS) -o test_rank2

test_copy_gpu:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_copy_gpu.cpp  $(LIB_SIRIUS) $(LIBS) -o test_copy_gpu

test_gen_eig: test_gen_eig.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) test_gen_eig.cpp  $(LIB_SIRIUS) $(LIBS) -o test_gen_eig

test_gpudirect: test_gpudirect.cpp
	$(CXX) $(CXX_OPT) $(INCLUDE) test_gpudirect.cpp  $(LIB_SIRIUS) $(LIBS) -o test_gpudirect

test_allreduce: test_allreduce.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) test_allreduce.cpp $(LIB_SIRIUS) $(LIBS) -o test_allreduce

test_linalg: test_linalg.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) test_linalg.cpp $(LIB_SIRIUS) $(LIBS) -o $@

test_bcast: test_bcast.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_zgemm_in_hpsi: test_zgemm_in_hpsi.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_alltoall: test_alltoall.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_transpose: test_transpose.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_fft: test_fft.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_xc: test_xc.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_enu: test_enu.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_bessel: test_bessel.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_spline: test_spline.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_symmetry: test_symmetry.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_roundoff: test_roundoff.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

clean:
	rm -rf *.o *.h5 *.txt *.dat *.pdf timers.json out.json splindex hdf5 hydrogen read_atom \
	fft fft1k spline allgather cuda_zgemm mt_function mt_kinetic spline_gpu fft_t test_mdarray \
	test_pstdout test_zgemm test_init test_blacs test_enu test_allreduce test_alltoall test_bcast \
	test_copy_gpu test_diag *dSYM

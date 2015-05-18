include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

all: hdf5 allgather mt_function mt_kinetic spline splindex hydrogen read_atom fft fft_t test_mdarray test_xc

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

###test_mdarray: test_mdarray.cpp $(LIB_SIRIUS)
###	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##hdf5: hdf5.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##allgather: allgather.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##mt_function: mt_function.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##mt_kinetic: mt_kinetic.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##spline: spline.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_zgemm: test_zgemm.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##splindex: splindex.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##hydrogen: hydrogen.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##fft: fft.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##fft_t: fft_t.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_pstdout: test_pstdout.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_init: test_init.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_init_global: test_init_global.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_blacs: test_blacs.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_iallgather: test_iallgather.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_diag: test_diag.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_rank2: test_rank2.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_copy_gpu: test_copy_gpu.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_gen_eig: test_gen_eig.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_gpudirect: test_gpudirect.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_allreduce: test_allreduce.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_linalg: test_linalg.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##	
##read_atom: read_atom.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_bcast: test_bcast.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_zgemm_in_hpsi: test_zgemm_in_hpsi.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_alltoall: test_alltoall.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_transpose: test_transpose.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_fft: test_fft.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_xc: test_xc.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_enu: test_enu.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_bessel: test_bessel.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_spline: test_spline.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_symmetry: test_symmetry.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@
##
##test_roundoff: test_roundoff.cpp $(LIB_SIRIUS)
##	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

clean:
	rm -rf *.o *.h5 *.txt *.dat *.pdf timers.json out.json splindex hdf5 hydrogen read_atom \
	fft fft1k spline allgather cuda_zgemm mt_function mt_kinetic spline_gpu fft_t test_mdarray \
	test_pstdout test_zgemm test_init test_blacs test_enu test_allreduce test_alltoall test_bcast \
	test_copy_gpu test_diag *dSYM

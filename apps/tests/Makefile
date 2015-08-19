include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

all: hdf5 allgather mt_function mt_kinetic spline splindex hydrogen read_atom fft fft_t \
     test_mdarray test_xc test_gemm

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_gemm: test_gemm.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -D_TEST_REAL_ $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_dgemm
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_zgemm

clean:
	rm -rf *.o *.h5 *.txt *.dat *.pdf timers.json out.json splindex hdf5 hydrogen read_atom \
	fft fft1k spline allgather cuda_zgemm mt_function mt_kinetic spline_gpu fft_t test_mdarray \
	test_pstdout test_zgemm test_init test_blacs test_enu test_allreduce test_alltoall test_bcast \
	test_copy_gpu test_diag *dSYM

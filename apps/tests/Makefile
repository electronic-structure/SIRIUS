include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

LIB_SIRIUS_DBG = ../../src/libsirius.dbg.a

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(CXX_OPT) -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

all: test_hdf5 test_allgather mt_function mt_kinetic splindex hydrogen read_atom \
     test_mdarray test_xc test_hloc test_mpi_grid test_mixer test_enu test_gemm

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_gemm: test_gemm.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -D__TEST_REAL -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_dgemm
	$(CXX) $(CXX_OPT) -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_zgemm

test_mpi_omp: test_mpi_omp.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -c kiss_fft.c -o kiss_fft.o
	$(CXX) $(CXX_OPT) -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) kiss_fft.o -o test_mpi_omp

clean:
	rm -rf *.o *.h5 *.txt *.dat *.pdf *dSYM timers.json out.json splindex test_hdf5 hydrogen read_atom \
	fft fft1k spline test_allgather cuda_zgemm mt_function mt_kinetic spline_gpu fft_t test_mdarray \
	test_pstdout test_zgemm test_init test_blacs test_enu test_allreduce test_alltoall test_bcast \
	test_copy_gpu test_diag *dSYM test_xc test_dgemm test_zgemm test_hloc test_complex_exp \
	test_fft_correctness test_memop test_mixer test_mpi_grid test_mutable test_sht test_splne \
	test_transpose test_spline test_transpose test_unit_cell

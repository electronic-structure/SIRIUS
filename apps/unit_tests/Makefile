include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

all: test_init test_sht test_fft_correctness test_spline

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

test_gemm: test_gemm.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -D_TEST_REAL_ $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_dgemm
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o test_zgemm

test_mpi_omp: test_mpi_omp.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -c kiss_fft.c -o kiss_fft.o
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) kiss_fft.o -o test_mpi_omp

clean:
	rm -rf *.o *.h5 *.txt *.dat *.pdf timers.json out.json test_init test_sht \
    test_fft_correctness test_spline

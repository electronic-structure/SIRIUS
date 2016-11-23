include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

.cpp.o:
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

all: test_init test_sht test_fft_correctness test_fft_real test_spline test_rot_ylm test_linalg test_wf_ortho

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

clean:
	rm -rf *.o test_init test_sht test_fft_correctness test_fft_real test_spline test_rot_ylm test_linalg test_wf_ortho *.dSYM

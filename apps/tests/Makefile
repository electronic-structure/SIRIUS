include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: hdf5 allgather mt_function mt_kinetic spline splindex hydrogen read_atom fft fft_t test_mdarray

hdf5:
	$(CXX) $(CXX_OPT) $(INCLUDE) hdf5.cpp           $(LIB_SIRIUS) $(LIBS) -o hdf5
	
allgather:
	$(CXX) $(CXX_OPT) $(INCLUDE) allgather.cpp      $(LIB_SIRIUS) $(LIBS) -o allgather
	
mt_function:
	$(CXX) $(CXX_OPT) $(INCLUDE) mt_function.cpp    $(LIB_SIRIUS) $(LIBS) -o mt_function
	
mt_kinetic:
	$(CXX) $(CXX_OPT) $(INCLUDE) mt_kinetic.cpp    $(LIB_SIRIUS) $(LIBS) -o mt_kinetic
	
spline:
	$(CXX) $(CXX_OPT) $(INCLUDE) spline.cpp        $(LIB_SIRIUS) $(LIBS) -o spline

test_zgemm:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_zgemm.cpp    $(LIB_SIRIUS) $(LIBS) -o test_zgemm
	
spline_gpu:
#	$(CXX) $(CXX_OPT) $(INCLUDE) spline_gpu.cpp    $(LIB_SIRIUS) $(LIBS) -o spline_gpu
	
splindex:
	$(CXX) $(CXX_OPT) $(INCLUDE) splindex.cpp      $(LIB_SIRIUS) $(LIBS) -o splindex
	
hydrogen:
	$(CXX) $(CXX_OPT) $(INCLUDE) hydrogen.cpp      $(LIB_SIRIUS) $(LIBS) -o hydrogen
	
read_atom:
	$(CXX) $(CXX_OPT) $(INCLUDE) read_atom.cpp     $(LIB_SIRIUS) $(LIBS) -o read_atom
	
fft:
	$(CXX) $(CXX_OPT) $(INCLUDE) fft.cpp           $(LIB_SIRIUS) $(LIBS) -o fft
	
fft_t:
	$(CXX) $(CXX_OPT) $(INCLUDE) fft_t.cpp         $(LIB_SIRIUS) $(LIBS) -o fft_t
	
test_mdarray:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_mdarray.cpp  $(LIB_SIRIUS) $(LIBS) -o test_mdarray

test_pstdout:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_pstdout.cpp  $(LIB_SIRIUS) $(LIBS) -o test_pstdout

test_init:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_init.cpp  $(LIB_SIRIUS) $(LIBS) -o test_init

test_blacs:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_blacs.cpp  $(LIB_SIRIUS) $(LIBS) -o test_blacs

test_iallgather:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_iallgather.cpp  $(LIB_SIRIUS) $(LIBS) -o test_iallgather

test_enu:
	$(CXX) $(CXX_OPT) $(INCLUDE) test_enu.cpp  $(LIB_SIRIUS) $(LIBS) -o test_enu
	

clean:
	rm -f *.o *.h5 *.txt *.dat *.pdf timers.json out.json splindex hdf5 hydrogen read_atom \
	fft fft1k spline allgather cuda_zgemm mt_function mt_kinetic spline_gpu fft_t test_mdarray \
	test_pstdout test_zgemm test_init test_blacs test_enu

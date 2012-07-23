include ../../make.inc

.SUFFIXES: .cpp .cu

NVCC = nvcc

SRC = ../libjson/Source/internalJSONNode.cpp ../libjson/Source/JSONAllocator.cpp ../libjson/Source/JSON_Base64.cpp \
      ../libjson/Source/JSONChildren.cpp ../libjson/Source/JSONDebug.cpp ../libjson/Source/JSONIterators.cpp \
      ../libjson/Source/JSONMemory.cpp ../libjson/Source/JSONNode.cpp ../libjson/Source/JSONNode_Mutex.cpp \
      ../libjson/Source/JSONPreparse.cpp ../libjson/Source/JSONStream.cpp ../libjson/Source/JSONValidator.cpp \
      ../libjson/Source/JSONWorker.cpp ../libjson/Source/JSONWriter.cpp ../libjson/Source/libjson.cpp \
      linalg_cpu.cpp sirius.cpp
      
OBJ = $(SRC:.cpp=.o) 
      
ifneq (,$(findstring D_GPU_,$(CPP_OPT)))
	OBJ := $(OBJ) gpu_interface.o
endif
	
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ -I/project/s299/magma/magma-1.1_my-xk6-gcc-4.6.2-libsci/include -c $<

all: $(OBJ)
	ar -r libsirius.a $(OBJ)

clean:
	rm $(OBJ) *.a

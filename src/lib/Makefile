include ../../make.inc

.SUFFIXES: .cpp .cu

NVCC = nvcc

SRC = ../libjson/_internal/Source/internalJSONNode.cpp ../libjson/_internal/Source/JSONAllocator.cpp \
      ../libjson/_internal/Source/JSONChildren.cpp ../libjson/_internal/Source/JSONDebug.cpp \
      ../libjson/_internal/Source/JSONIterators.cpp ../libjson/_internal/Source/JSONMemory.cpp \
      ../libjson/_internal/Source/JSONNode.cpp ../libjson/_internal/Source/JSONNode_Mutex.cpp \
      ../libjson/_internal/Source/JSONPreparse.cpp ../libjson/_internal/Source/JSONStream.cpp \
      ../libjson/_internal/Source/JSONValidator.cpp ../libjson/_internal/Source/JSONWorker.cpp \
      ../libjson/_internal/Source/JSONWriter.cpp ../libjson/_internal/Source/libjson.cpp \
      sirius.cpp LebedevLaikov.cpp
      
OBJ = $(SRC:.cpp=.o) 
      
ifneq (,$(findstring D_GPU_,$(CPP_OPT)))
	OBJ := $(OBJ) gpu_interface.o
endif
	
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

.cu.o:
	$(NVCC) -DADD_ -I/project/s299/magma/magma-1.1_my-xk6-gcc-4.6.2-libsci/include -c $<

all: log $(OBJ)
	ar -r libsirius.a $(OBJ)

# TODO: platform-independent python script
log:
	@echo "const char* git_hash = \"$(shell git rev-parse HEAD)\";" > version.h 
	@echo "const char* build_date = \"$(shell date)\";" >> version.h 

clean:
	rm $(OBJ) *.a

depend: $(SRC)
	rm -f ./.depend
	$(CXX) $(CXX_OPT) -MM $^ >> ./.depend

showlibs:
	@echo $(LIBS) $(shell pwd)/libsirius.a 

include .depend

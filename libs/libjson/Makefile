include ../../make.inc

.SUFFIXES: .cpp

SRC = ./_internal/Source/internalJSONNode.cpp ./_internal/Source/JSONAllocator.cpp \
      ./_internal/Source/JSONChildren.cpp     ./_internal/Source/JSONDebug.cpp \
      ./_internal/Source/JSONIterators.cpp    ./_internal/Source/JSONMemory.cpp \
      ./_internal/Source/JSONNode.cpp         ./_internal/Source/JSONNode_Mutex.cpp \
      ./_internal/Source/JSONPreparse.cpp     ./_internal/Source/JSONStream.cpp \
      ./_internal/Source/JSONValidator.cpp    ./_internal/Source/JSONWorker.cpp \
      ./_internal/Source/JSONWriter.cpp       ./_internal/Source/libjson.cpp
      
OBJ = $(SRC:.cpp=.o) 
      
.cpp.o:
	$(CXX) $(CXX_OPT) -c -o $(*D)/$(*F).o $<

all: $(OBJ)
	ar -r libjson.a $(OBJ)

clean:
	rm $(OBJ) *.a


ARCH=--generate-code arch=compute_35,code=sm_35 \
     --generate-code arch=compute_35,code=sm_37 \
     --generate-code arch=compute_50,code=sm_50 \
     --generate-code arch=compute_50,code=sm_52 \
     --generate-code arch=compute_53,code=sm_53 \
     --generate-code arch=compute_60,code=sm_60 \
     --generate-code arch=compute_61,code=sm_61 \
     --generate-code arch=compute_62,code=sm_62 \
     --generate-code arch=compute_70,code=sm_70 \
     --generate-code arch=compute_72,code=sm_72 \
     --generate-code arch=compute_75,code=[sm_75,compute_75]

RELEASE=cufathon

OBJ_DIR=./obj/

COMPILER=g++
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS=-lm -lrt -lstdc++fs
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -Wunused -Wunreachable-code -Wno-deprecated-declarations
CFLAGS+=$(OPTS)

CFLAGS_GPU=$(CFLAGS)
CFLAGS_CPU=$(CFLAGS)
CFLAGS_CPU+=-std=c++17

COMMON+=-I/usr/local/cuda/include/
LDFLAGS+=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand

OBJ_MAIN=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJ_CPU=$(patsubst %.cpp,%.o,$(wildcard c++/*.cpp))
OBJ_GPU=$(patsubst %.cu,%.o,$(wildcard cuda/*.cu))

OBJ_TOT=$(OBJ_MAIN) $(OBJ_CPU) $(OBJ_GPU)

OBJS=$(addprefix $(OBJ_DIR), $(OBJ_TOT))

DEPS_CPU=$(wildcard c++/*.h)
DEPS_GPU=$(wildcard cuda/*.cuh)

all: release

.PHONY: release

release: obj $(RELEASE)

obj:
	mkdir -p ./obj/c++
	mkdir -p ./obj/cuda

$(RELEASE): $(OBJS)
	$(COMPILER) $(COMMON) $(CFLAGS_CPU) $^ -o $@ $(LDFLAGS)
	
$(OBJ_DIR)%.o: %.cpp $(DEPS_CPU)
	$(COMPILER) $(COMMON) $(CFLAGS_CPU) -c $< -o $@ $(LDFLAGS)

$(OBJ_DIR)%.o: %.cu $(DEPS_GPU)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS_GPU)" -c $< -o $@

.PHONY: clean

clean:
	rm -rf ./obj/

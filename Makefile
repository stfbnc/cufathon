RELEASE=cufathon

CPP=g++
NVCC=nvcc 
OPTS=-Ofast -std=c++17

INCS+=-I/usr/local/cuda/include/
LIBS+=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand

.PHONY: all

all: $(RELEASE)

$(RELEASE): dfa_kernel.o main.o
	$(CPP) -o $(RELEASE) dfa_kernel.o main.o $(LIBS)

dfa_kernel.o: ./cuda/dfa_kernel.cu ./cuda/dfa_kernel.cuh
	$(NVCC) $(INCS) -c ./cuda/dfa_kernel.cu
main.o: main.cpp
	$(CPP) $(INCS) $(OPTS) -c main.cpp

.PHONY: clean

clean:
	rm -f ./*.o
	rm -f $(RELEASE)

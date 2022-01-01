RELEASE=cufathon

CPP=g++
NVCC=nvcc 
OPTS=-Ofast -std=c++17

INCS+=-I/usr/local/cuda/include/
LIBS+=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand

.PHONY: all

all: $(RELEASE)

$(RELEASE): main.o dfa_kernel.o mfdfa_kernel.o dcca_kernel.o ht_kernel.o main.o
	$(CPP) -o $(RELEASE) dfa_kernel.o mfdfa_kernel.o dcca_kernel.o ht_kernel.o main.o $(LIBS)

dfa_kernel.o: ./cuda/dfa_kernel.cu ./cuda/dfa_kernel.cuh
	$(NVCC) $(INCS) -c ./cuda/dfa_kernel.cu
mfdfa_kernel.o: ./cuda/mfdfa_kernel.cu ./cuda/mfdfa_kernel.cuh
	$(NVCC) $(INCS) -c ./cuda/mfdfa_kernel.cu
dcca_kernel.o: ./cuda/dcca_kernel.cu ./cuda/dcca_kernel.cuh
	$(NVCC) $(INCS) -c ./cuda/dcca_kernel.cu
ht_kernel.o: ./cuda/ht_kernel.cu ./cuda/ht_kernel.cuh
	$(NVCC) $(INCS) -c ./cuda/ht_kernel.cu
main.o: main.cpp
	$(CPP) $(INCS) $(OPTS) -c main.cpp

.PHONY: clean

clean:
	rm -f ./*.o
	rm -f $(RELEASE)

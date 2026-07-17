NVCC = nvcc
CFLAGS = -O3

TARGET = attention_test

all: $(TARGET)

$(TARGET): main.o kernels.o
	$(NVCC) $^ -o $@

main.o: main.cu kernels.h
	$(NVCC) $(CFLAGS) -c main.cu

kernels.o: kernels.cu kernels.h
	$(NVCC) $(CFLAGS) -c kernels.cu

clean:
	rm -f *.o $(TARGET)
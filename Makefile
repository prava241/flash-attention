NVCC = nvcc

CFLAGS = -O3

TARGET = attention

SRC = \
src/main.cpp \
src/attention.cu \
src/cpu_reference.cpp

all:
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
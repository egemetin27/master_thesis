# Makefile for building orchestrator
# Usage: make
# Optional: make clean

CXX := g++
CXXFLAGS := -std=c++20 -I/usr/local/cuda-12.8/include
LDFLAGS := -L/usr/local/cuda-12.8/lib64
LDLIBS := /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 -lpthread

.PHONY: all clean

all: orchestrator

orchestrator: orchestrator.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

run: 
	./orchestrator

trace: 
	./orchestrator --trace

cleanRun: orchestrator
	./orchestrator

clean:
	rm -f orchestrator

debugnvml: debugnvml.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)
	./debugnvml

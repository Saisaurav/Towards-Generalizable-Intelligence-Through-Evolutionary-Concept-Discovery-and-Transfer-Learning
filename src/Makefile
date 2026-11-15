CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -pthread -Wall -Wextra
VERSION := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet 2>/dev/null || echo -dirty)
all: brain runner
brain: main.cpp neat.h ppo.h net.h genome.h env.h envs.h rng.h concept_env.h external_env.h
	$(CXX) $(CXXFLAGS) main.cpp -o brain
runner: runner.cpp neat.h ppo.h net.h genome.h env.h envs.h rng.h concept_env.h wrappers.h
	$(CXX) $(CXXFLAGS) -DREPO_VERSION='"$(VERSION)"' runner.cpp -o runner
regression_test: regression_test.cpp neat.h ppo.h net.h genome.h envs.h env.h rng.h concept_env.h
	$(CXX) $(CXXFLAGS) regression_test.cpp -o regression_test
env_smoke_test: env_smoke_test.cpp envs.h env.h rng.h concept_env.h
	$(CXX) $(CXXFLAGS) env_smoke_test.cpp -o env_smoke_test
lamarckian_hidden_test: lamarckian_hidden_test.cpp neat.h ppo.h net.h genome.h envs.h env.h rng.h concept_env.h
	$(CXX) $(CXXFLAGS) lamarckian_hidden_test.cpp -o lamarckian_hidden_test
test-envs: env_smoke_test lamarckian_hidden_test regression_test
	./env_smoke_test
	./lamarckian_hidden_test
	./regression_test
clean:
	rm -f brain runner regression_test env_smoke_test lamarckian_hidden_test *.csv
.PHONY: all clean test-envs

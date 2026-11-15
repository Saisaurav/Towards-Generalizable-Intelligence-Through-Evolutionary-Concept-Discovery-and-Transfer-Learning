CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -pthread -Wall -Wextra
LDFLAGS ?=
ifeq ($(OS),Windows_NT)
LDFLAGS += -static
endif
VERSION := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet 2>/dev/null || echo -dirty)
all: brain runner
brain: main.cpp neat.h ppo.h net.h genome.h env.h envs.h rng.h concept_env.h external_env.h fixed_head_env.h robotics_env.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) main.cpp -o brain
runner: runner.cpp neat.h ppo.h net.h genome.h env.h envs.h rng.h concept_env.h wrappers.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -DREPO_VERSION='"$(VERSION)"' runner.cpp -o runner
regression_test: regression_test.cpp neat.h ppo.h net.h genome.h envs.h env.h rng.h concept_env.h fixed_head_env.h robotics_env.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) regression_test.cpp -o regression_test
learner_integration_test: learner_integration_test.cpp neat.h ppo.h net.h genome.h env.h rng.h concept_env.h fixed_head_env.h robotics_env.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) learner_integration_test.cpp -o learner_integration_test
env_smoke_test: env_smoke_test.cpp envs.h env.h rng.h concept_env.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) env_smoke_test.cpp -o env_smoke_test
test-envs: env_smoke_test regression_test learner_integration_test
	./env_smoke_test
	./regression_test
	./learner_integration_test
clean:
	rm -f brain runner regression_test env_smoke_test learner_integration_test *.csv
.PHONY: all clean test-envs

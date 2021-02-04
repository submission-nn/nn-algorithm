#include <stdint.h>
#include <sys/mman.h>

#define G_n 250
constexpr uint64_t w = 0.2*G_n;
constexpr uint64_t r = 10;
constexpr uint64_t d = uint64_t(0.3*G_n);
constexpr uint64_t N = 10000;
constexpr uint64_t epsilon = 1;

constexpr bool find_all = true;
constexpr uint64_t THRESHHOLD = 500;

#include "helper.h"
#include "container.h"
#include "list.h"
#include "nn.h"
#include "windowed_nn_v2.h"

#include "b63.h"
#include "counters/perf_events.h"

#define TEST_BASE_LIST_SIZE (1u << 10u)

#define RUNTIME_LIMITER(n) 1


B63_BASELINE(Quadratic, n) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	B63_SUSPEND {
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
	}

	NearestNeighbor nn{L1, L2, w, r, N, d, find_all, THRESHHOLD};
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	int32_t res = 0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN(gold1, gold2);
		res += found;
	}

	B63_SUSPEND {res += 1; }
	B63_KEEP(res);
}


B63_BENCHMARK(Windowed2, n) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	B63_SUSPEND {
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, w/r);
	}

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD};
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	int32_t res = 0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN(gold1, gold2);
		res += found;
	}

	B63_SUSPEND {res += 1; }
	B63_KEEP(res);
}

B63_BENCHMARK(Windowed2Epsilon, n) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	B63_SUSPEND {
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, w/r);
	}

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	int32_t res = 0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN(gold1, gold2);
		res += found;
	}

	B63_SUSPEND {res += 1; }
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
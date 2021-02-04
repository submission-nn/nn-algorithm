#include <stdint.h>

#define G_n        1000u
constexpr uint64_t w = 0.15*G_n;
constexpr uint64_t r = 10;
constexpr uint64_t d = uint64_t(0.2*G_n);
constexpr uint64_t N = 10000;
constexpr uint64_t epsilon = 1;

constexpr bool find_all = true;
constexpr uint64_t THRESHHOLD = 500;

#define TEST_BASE_LIST_SIZE (1u << 15u)

// If you uncomment the next line, one will see more debugging outout. Usfull if one ones to check the correctness.
//#define EXT_LOG(x) x
#define EXT_LOG(x)


#include "helper.h"
#include "container.h"
#include "list.h"
#include "nn.h"
#include "windowed_nn_v2.h"

#include "b63.h"
#include "counters/perf_events.h"



#define RUNTIME_LIMITER(n) 1


B63_BASELINE(Quadratic, n) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	B63_SUSPEND {
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
	}

	NearestNeighbor nn{L1, L2, w, r, N, d, find_all, THRESHHOLD};

	int32_t res = 0, errc = 0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN();
		if (unlikely((found == 0) && (nn.sols_1.empty())) )
			errc += 1;

		res +=  found;
	}

	// How many solutions where found.
	EXT_LOG(std::cout << "err quad: " << n << " " << errc << " " << double(errc)/double(n) << "\n";)

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

	int32_t res = 0, errc=0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN();
		if (unlikely((found == 0) && (nn.sols_1.empty())) )
			errc += 1;

		res += found;
	}

	// How many solutions where found.
	EXT_LOG(std::cout << "err win2: " << n << " " << errc << " " << double(errc)/double(n) << "\n";)

	B63_SUSPEND { res += 1; }
	B63_KEEP(res);
}

B63_BENCHMARK(Windowed2Epsilon, n) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	B63_SUSPEND {
		NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2, true, w/r);
	}

	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};

	int32_t res = 0, errc=0;

	for (int i = 0; i < RUNTIME_LIMITER(n); i++) {
		uint64_t found = nn.NN();
		if (unlikely((found == 0) && (nn.sols_1.empty())) )
			errc += 1;

		res += found;
	}

	// How many solutions where found.
	EXT_LOG(std::cout << "err win2: " << n << " " << errc << " " << double(errc)/double(n) << "\n";)

	B63_SUSPEND { res += 1; }
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
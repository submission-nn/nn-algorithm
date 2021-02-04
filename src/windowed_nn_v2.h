#ifndef SMALLSECRETLWE_WNN2_H
#define SMALLSECRETLWE_WNN2_H

#include <sys/mman.h>

#include "helper.h"
#include "container.h"
#include "list.h"
#include "nn.h"

class WindowedNearestNeighbor2 : public NearestNeighbor {

	// Additional parameters which are not defined in the base class.
	const double d_;        // delta/n
	const uint64_t k;       // n/r BlockSize
	const uint64_t dk;      // weight per block
	const uint64_t epsilon; // additional offset/variance we allow in each level to mach on.

	// Array indicating the window boundaries.
	std::vector<uint64_t> buckets_windows{};

	// How many solution did we already searched in the golden NN search.
	uint64_t solution_searched = 0;

	// translate a block level into actual bit limits.
	constexpr void get_window(uint64_t *k_lower, uint64_t *k_higher, const uint64_t current_r){
		// Only Useful during debugging. Its equal to: Match on all coordinates.
		if (current_r == -1){
			*k_lower = 0;
			*k_higher = n;
			return;
		}

		ASSERT(current_r < buckets_windows.size());
		*k_lower = buckets_windows[current_r];
		*k_higher = buckets_windows[current_r+1];
	}

public:
	WindowedNearestNeighbor2(NNList &L1, NNList &L2, const uint64_t w, const uint64_t r, const uint64_t N, const uint64_t d,
	                         const bool find_all=true, const uint64_t THRESHHOLD=500, const uint64_t epsilon=0):
	                         d_(double(d)/double(n)), k(n/r), dk(d_*double(n)/double(r)), epsilon(epsilon),
	                         NearestNeighbor(L1, L2, w, r, d, N, find_all, THRESHHOLD) {
		ASSERT(THRESHHOLD > 0 && dk-epsilon > 0 && r > 0);

		// initialize the windows.
		buckets_windows.resize(r+1, 0);
		for (int i = 0; i < r; ++i) {
			buckets_windows[i] = i*k;
		}
		buckets_windows[r] = n;

#ifdef DEBUG
		//std::cout << "q:" << q << "\n";
		std::cout << "w:" << w << "\n";
		std::cout << "N:" << N << "\n";
		std::cout << "number of buckets r:" << r << "\n";
		std::cout << "d:" << d << "\n";
		std::cout << "d_:" << d_ << "\n";
		std::cout << "bucket length k:" << k << "\n";
		std::cout << "weight per bucket dk:" << dk << "\n";

		std::cout << "bucket windows: \n";
		for (const auto i : buckets_windows)
			std::cout << i << " ";
		std::cout << "\n";
#endif
	}

	void windowed_SortList(NNList &L, const NNContainer &z,
	                       const uint64_t k_lower, const uint64_t k_upper,
	                       const uint64_t start, const uint64_t end,
	                       uint64_t &new_start, uint64_t &new_end) {
		ASSERT(L.size() > 0 && end <= L.size() && start < end && k_lower < k_upper && k_upper <= n);

#ifdef __unix__
		// make sure that the kernel knows how we want to access the memory.
		madvise(L1.data(), L1.size()* NNContainer::get_size()/8, POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);
		madvise(L2.data(), L1.size()* NNContainer::get_size()/8, POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);
#endif

		NNContainer tmp;
		new_start = start;  // hardcode it.
		uint64_t ctr = 0;   // how many elements did we found?

		for (uint64_t i = start; i < end; ++i) {
			auto b = NNContainer::add(tmp, L[i], z, k_lower, k_upper);
#if defined(ALL_DELTA)
			if ((b <= dk) && (ctr != i)){
#else
			if ((b >= dk-epsilon) && (b <= dk+epsilon) && (ctr != i)){
#endif
				tmp = L[start+ctr];
				L[start+ctr] = L[i];
				L[i] = tmp;

				ctr += 1;
			}
		}
		if (ctr != 0)
			ctr -= 1;

		new_end = ctr;
	}

	int NN() {
		// Reset everything.
		sols_1.resize(0);
		sols_2.resize(0);

		bool abort;
		uint64_t k_lower, k_upper;
		get_window(&k_lower, &k_upper, 0);

		for (int i = 0; i < N; ++i) {
			abort = NN_internal(0, L1.size(), 0, L2.size(), 0, k_lower, k_upper, nullptr, nullptr);
			if (abort)
				return abort;
		}

		// kek nothing found
		return 0;
	}

	// perform NN search until the golden solution is found
	int NN(const NNContainer &gold1, const NNContainer &gold2) {
		sols_1.resize(0);
		sols_2.resize(0);

		bool not_found = true;
		int i;

		uint64_t k_lower, k_upper;
		get_window(&k_lower, &k_upper, 0);

		while (not_found)  {
			NN_internal(0, L1.size(), 0, L2.size(), 0, k_lower, k_upper, &gold1, &gold2);
			ASSERT(sols_1.size() == sols_2.size());

			for (i = 0; i < sols_1.size(); ++i) {
				if (gold1.is_equal(sols_1[i]) && gold2.is_equal(sols_2[i])){
					not_found = false;
					break;
				}
			}
		}

		// we dont want that this function is optimized out in synthetic benchmarks
		return i;
	}

	// perform NN Search until we have found 'nr_solutions'
	int NN(const uint64_t nr_solutions) {
		sols_1.resize(0);
		sols_2.resize(0);

		uint64_t k_lower, k_upper;
		get_window(&k_lower, &k_upper, 0);

		uint64_t perf_counter = 0;

		while (true)  {
			NN_internal(0, L1.size(), 0, L2.size(), 0, k_lower, k_upper, nullptr, nullptr);
			ASSERT(sols_1.size() == sols_2.size());

			perf_counter += 1;
			if (sols_1.size() >= nr_solutions){
				return nr_solutions;
			}
		}
	}
private:
	// Some performance logging:
#ifdef PERFOMANCE_LOGGING
	uint64_t max_tree_depth = 0;
	uint64_t new_zero_tables = 0;
	uint64_t non_golden_found = 0;
#endif

	// TODO remove
	bool flag = true;
	int NN_internal(const uint64_t start1, const uint64_t end1,
	                          const uint64_t start2, const uint64_t end2,
	                          uint64_t current_r,
	                          const uint64_t k_lower, const uint64_t k_upper,
	                          const NNContainer *gold1, const NNContainer *gold2) {
		ASSERT(start1 < end1 && start2 < end2 && k_lower < k_upper && k_upper <= n);

		// first init
		NNContainer z;
		z.random();

		uint64_t new_start1, new_end1, new_start2, new_end2;
		windowed_SortList(L1, z, k_lower, k_upper, start1, end1, new_start1, new_end1);
		windowed_SortList(L2, z, k_lower, k_upper, start2, end2, new_start2, new_end2);

#ifdef LOGGING
		// some helpful logging
		logging_list(L1, z, start1, end1, k_lower, k_upper);
		logging_list(L2, z, start2, end2, k_lower, k_upper);
		logging(z, new_start1, new_end1, new_start2, new_end2, k_lower, k_upper);
#endif
#ifdef PERFOMANCE_LOGGING
		if (current_r > max_tree_depth)
			max_tree_depth = current_r;

		if (((new_end1 - new_start1) == 0) || ((new_end2 - new_start2) == 0))
			new_zero_tables += 1;
#endif

		// No possible solution survived
		if (((new_end1 - new_start1) == 0) || ((new_end2 - new_start2) == 0))
			return 0;

		// fallback to naive NN algorithm if threshold is reached.
		if (((new_end1 - new_start1) < THRESHHOLD) || ((new_end2 - new_start2) < THRESHHOLD)){
			return quadratic_NN(new_start1, new_end1, new_start2, new_end2, gold1, gold2);
		}

		// max depth is reached. Switch to naive search
		if (current_r == r-1){
			return quadratic_NN(new_start1, new_end1, new_start2, new_end2, gold1, gold2);
		}

		int abort;

		const uint64_t new_current_r = current_r + 1;
		uint64_t new_k_lower, new_k_upper;
		get_window(&new_k_lower, &new_k_upper, new_current_r);

		for (int i = 0; i < N; ++i) {
			abort = NN_internal(new_start1, new_end1,
								        new_start2, new_end2, new_current_r,
								        new_k_lower, new_k_upper,
								        gold1, gold2);
			if (abort > 0) {
				if (gold1 != nullptr && gold2 != nullptr) {
					for (int j = solution_searched; j < sols_1.size(); ++j) {
						if (gold1->is_equal(sols_1[j]) && gold2->is_equal(sols_2[j])){
							return j;
						}
					}
#ifdef PERFOMANCE_LOGGING
					else {
						non_golden_found += 1;
					}
#endif
					solution_searched = sols_1.size();
				} else {
					return abort;
				}
			}
		}

		// nothing found
		return 0;
	}

	void information()  { std::cout << "WindowedNN2:\n"; }
};

#endif //SMALLSECRETLWE_WNN2_H

#ifndef NN_CODE_INCLUDE_H
#define NN_CODE_INCLUDE_H

#include "container.h"
#include "list.h"
#include "windowed_nn_v2.h"
#include "indyk_motwani.h"

#ifndef ITERS
#define ITERS   1
#endif
#ifndef RUNS
#define RUNS    50
#endif


#define LOGGING
//#define CSV_LOGGING
// #define SOLUTION_LOGGING

void logg(const std::string &alg, const double time, const double sols = 0.0) {
#ifdef LOGGING
	// normal human readable logging
	std::cout << alg <<" time: " << time << "s";
	if (sols == 0.0) {
		std::cout << "\n" << std::flush;
	}else {
		std::cout << " sols: " << sols << "\n"<< std::flush;
	}
#elif defined(CSV_LOGGING)
	// csv logging.
	std::cout << alg << "," << time << "\n";
#endif
}

static void run(const uint64_t w, const uint64_t d, const uint64_t r, const uint64_t N, const uint64_t size,
				const bool find_all, const uint64_t THRESHHOLD, const uint64_t epsilon,
				const std::string &alg, double *ret_time= nullptr) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0;
	for (int i = 0; i < ITERS; ++i) {
		double time = 0;
		NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
			uint64_t found = nn.NN(gold1, gold2);
			double t1 = ((double)clock()/CLOCKS_PER_SEC) - t0;
			time += t1;

			EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
			EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
			EXPECT_EQ(nn.print_result(gold1, gold2), true);
			EXPECT_EQ(L1.size(), size);
			EXPECT_EQ(L2.size(), size);
			logg(alg,  t1);

		}

		iter += time/RUNS;
	}

	logg(alg,  iter/ITERS);
	if(ret_time != nullptr)
		*ret_time = iter/ITERS;
}

static void run_indyk(const uint64_t w, const uint64_t l, const bool find_all, const uint64_t size,
                const std::string &alg, double *ret_time= nullptr) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0;
	for (int i = 0; i < ITERS; ++i) {
		double time = 0;
		NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			IndykMontwani nn{L1, L2, w, l};
			uint64_t found = nn.indyk_NN(gold1, gold2);
			double t1 = ((double)clock()/CLOCKS_PER_SEC) - t0;
			time += t1;

			EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
			EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
			EXPECT_EQ(nn.print_result(gold1, gold2), true);
			EXPECT_EQ(L1.size(), size);
			EXPECT_EQ(L2.size(), size);
			logg(alg,  t1);

		}

		iter += time/RUNS;
	}

	logg(alg,  iter/ITERS);
	if(ret_time != nullptr)
		*ret_time = iter/ITERS;
}


static void run_dist(const uint64_t gamma, const uint64_t w, const uint64_t d, const uint64_t r, const uint64_t N, const uint64_t size,
                const bool find_all, const uint64_t THRESHHOLD, const uint64_t epsilon,
                const std::string &alg) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0.0;
	double sols = 0.0;

	for (int i = 0; i < ITERS; ++i) {
		double time = 0.0;
		double inner_sols = 0.0;

		NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gamma, w, pos1, pos2, true, r);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
			uint64_t found = nn.NN(gold1, gold2);
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;

			EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
			EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
			EXPECT_EQ(nn.print_result(gold1, gold2), true);
			EXPECT_EQ(L1.size(), size);
			EXPECT_EQ(L2.size(), size);

			inner_sols += nn.sols_1.size();
		}

		iter += time/RUNS;
		sols += inner_sols/RUNS;
	}

	logg(alg,  iter/ITERS, sols/ITERS);
}


static void run_dist_ratio(const uint64_t gamma, const uint64_t w, const uint64_t d, const uint64_t r, const uint64_t N, const uint64_t size,
                     const bool find_all, const uint64_t THRESHHOLD, const uint64_t epsilon, const double ratio, const std::string &alg) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0.0;
	double sols = 0.0;

	NearestNeighbor::create_test_lists_with_distribution(L1, L2, size, gamma, w, pos1, pos2, true, r);
	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	NearestNeighbor nn_q{L1, L2, w, 1 , 1, 1, true};
	nn_q.NN(gold1, gold2);

	// EXPECT_EQ(nn_q.every_solution_unique(), true);

	const uint64_t nr_solutions = nn_q.sols_1.size();
	// std:: cout << "finished Naive Search: " << nr_solutions<< "\n" << std::flush;

	for (int i = 0; i < ITERS; ++i) {
		double time = 0.0;
		double inner_sols = 0.0;

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, THRESHHOLD, epsilon};
			uint64_t found = nn.NN(nr_solutions*ratio);
			time += ((double)clock()/CLOCKS_PER_SEC) - t0;

			EXPECT_EQ(nn.print_result(gold1, gold2), true);
			EXPECT_EQ(L1.size(), size);
			EXPECT_EQ(L2.size(), size);
			EXPECT_GE(found,  uint64_t(nr_solutions*ratio));

			inner_sols += nn.sols_1.size();
			
			// Next Check that each solution found by our algorithm is also in the list of solutions of the naive search.
			for (int k = 0; k < nn.sols_1.size(); ++k) {
				EXPECT_EQ(nn_q.already_found(nn.sols_1[i], nn.sols_2[i]), true);
			}
		}

		iter += time/RUNS;
		sols += inner_sols/RUNS;
	}

	logg(alg,  iter/ITERS, sols/ITERS);
}

static void run_quadratic(const uint64_t w, const uint64_t size, const std::string &alg, double *ret_time= nullptr) {
	NNList L1{1}, L2{1};
	uint64_t pos1, pos2;

	double iter = 0;
	for (int i = 0; i < ITERS; ++i) {
		double time = 0;
		NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2);
		const NNContainer gold1 = L1[pos1];
		const NNContainer gold2 = L2[pos2];

		for (int j = 0; j < RUNS; ++j) {
			double t0 = (double)clock()/CLOCKS_PER_SEC;

			NearestNeighbor nn{L1, L2, w, 0 , 0, 0, true, 0};
			uint64_t found = nn.NN();
			double t1 = ((double)clock()/CLOCKS_PER_SEC) - t0;
			time += t1;

			EXPECT_EQ(gold1.is_equal(nn.sols_1[found]), true);
			EXPECT_EQ(gold2.is_equal(nn.sols_2[found]), true);
			EXPECT_EQ(nn.print_result(gold1, gold2), true);
			EXPECT_EQ(L1.size(), size);
			EXPECT_EQ(L2.size(), size);
			logg(alg,  t1);

		}

		iter += time/RUNS;
	}

	logg(alg,  iter/ITERS);
	if(ret_time != nullptr)
		*ret_time = iter/ITERS;
}

#endif //NN_CODE_INCLUDE_H

#ifndef NN_CODE_WINDOWED_NN_H
#define NN_CODE_WINDOWED_NN_H

#include <cmath>            // for log2()
#include <cstdint>
#include "helper.h"
#include "container.h"
#include "list.h"

class NearestNeighbor {
protected:
	// Bit length of each binary Container
	constexpr static uint64_t n = NNContainer ::get_size();   // = n

	// Minimal List Size to switch to a naive NN algorithm
	const uint64_t THRESHHOLD;   // completely arbitrary.

	// find all possible solutions
	// if set to 'true' the naive NN search wont stop after a valid tuple is found.
	const bool find_all;

	// target lists. Only as references.
	NNList &L1;
	NNList &L2;

	// parameter set
	const uint64_t w;   // omega: weight difference of the two elements x, y
	const uint64_t r;   // recursion depth
	const uint64_t N;   // forking factor
	const uint64_t d;   // delta

	// How many solution did we already searched in the golden NN search.
	uint64_t solution_searched = 0;
public:
	NearestNeighbor(NNList &L1, NNList &L2, const uint64_t w, const uint64_t r, const uint64_t N, const uint64_t d,
	                const bool find_all=true, const uint64_t THRESHHOLD=500) :
			L1(L1), L2(L2), w(w), r(r), N(N), d(d),
			find_all(find_all), THRESHHOLD(THRESHHOLD) {}

protected:
	virtual void information() { std::cout << "NN:\n"; }

public:
	void logging_list(const NNList &L, const NNContainer &z, const uint64_t start, const uint64_t end,
	                  const uint64_t k_lower=0, const uint64_t k_higher=n) {
		NNContainer tmp;

		std::cout << "List(corrected): start " << start << " end: " << end << "\n";
		std::cout << "z: " << z << "\n";
		for (int i = start; i < end; ++i) {
			tmp = L[i];
			NNContainer::add(tmp, L[i], z, k_lower, k_higher);
			std::cout << i << ": ";
			print(tmp, k_lower, k_higher);
			std::cout << " w:" << tmp.weight(k_lower, k_higher) << "\n";
		}
	}

	void logging(const NNContainer &z, const uint64_t start1, const uint64_t end1,
	             const uint64_t start2, const uint64_t end2,
	             const uint64_t k_lower=0, const uint64_t k_higher=n) {
		constexpr uint64_t log_list_offset = 0;
		NNContainer tmp;

		std::cout << "w:     " << w << "\n";
		std::cout << "start1:" << start1 << "\n";
		std::cout << "end1:  " << end1 << "\n";
		std::cout << "start2:" << start2 << "\n";
		std::cout << "end2:  " << end2 << "\n";
		std::cout << "k_lower:" << k_lower << "\n";
		std::cout << "k_higher:  " << k_higher << "\n";
		std::cout << "z:  " << z << "\n";

		// also log some part of the list.
		std::cout << "Partial L1:\n";
		for (int i = MIN(start1 - log_list_offset, start1); i < MIN(end1 + log_list_offset, L1.size()); ++i) {
			tmp = L1[i];
			NNContainer::add(tmp, L1[i], z, k_lower, k_higher);
			print(tmp, k_lower, k_higher);
			std::cout << " w:" << tmp.weight(k_lower, k_higher) << " uncorrected: " << L1[i] << " w:"
			          << L1[i].weight(k_lower, k_higher) << "\n";
		}

		std::cout << "Partial L2:\n";
		for (int i = MIN(start2 - log_list_offset, start2); i < MIN(end2 + log_list_offset, L2.size()); ++i) {
			tmp = L2[i];
			NNContainer::add(tmp, L2[i], z, k_lower, k_higher);
			print(tmp, k_lower, k_higher);
			std::cout << " w:" << tmp.weight(k_lower, k_higher) << " uncorrected: " << L2[i] << " w:"
			          << L2[i].weight(k_lower, k_higher) << "\n";
		}

		std::cout << "\n";
	}

	// calculate the log to base 2
	inline static double log2(double in) { return log(in) / log(2.); }

	inline static double H(double in) {
		// binary entropy functions
		ASSERT(in <= 1. && in >= 0.);
		if ((in == 1.) || in == 0.)
			return 0.0;

		return -(in * log2(in) + (1 - in) * log2(1 - in));
	}

	inline static double H1(double in) {
		// approximate inverse binary entropy function
		// its actually quite precise
		double steps[] = {0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001};
		double r = 0.;

		for (double step : steps) {
			double i = r;
			for (; (i+step < 1.0) && (H(i) < in); i += step) {}
			r = i - step;
		}
		return r;
	}


	inline static double calc_q(const uint64_t w, const uint64_t d) {
		const auto   n_ = double(n);
		const double w_ = double(w) / n_;
		const double d_ = double(d) / n_;
		return calc_q(n, w_, d_);
	}

	inline static double calc_q(const uint64_t n, const double w, const double d) {
		const double wm1 = 1 - w;
		// because H(0.5) = 1 and from (1/2)^{-n} comes a -1 we can skip them.
		return wm1 * H((d - (w / 2)) / wm1);
	}

	inline static double calc_q2(const uint64_t d) {
		const auto   n_ = double(n);
		const double d_ = double(d) / n_;
		return H(d_) - 1.0;
	}


	// checks if a solutions is already found
	bool already_found(const NNContainer &a, const NNContainer &b) {
		ASSERT(sols_1.size() == sols_2.size());
		for (uint64_t i = 0; i < sols_1.size(); ++i) {
			if (a.is_equal(sols_1[i]) && b.is_equal(sols_2[i]))
				return true;
		}
		return false;
	}

	// checks if every solution is unique. Little helper for some special case benchmarking.
	// Do not call it on big lists. If so, this will possible break your computer.
	bool every_solution_unique() {
		ASSERT(sols_1.size() == sols_2.size());

		for (uint64_t i = 0; i < sols_1.size(); ++i) {
			for (uint64_t j = 0; j < sols_1.size(); ++j) {
				if (j == i)
					continue;

				if (sols_1[i].is_equal(sols_1[j]) && sols_2[i].is_equal(sols_2[j]))
					return false;
			}
		}

		return true;
	}
public:
	std::vector<NNContainer> sols_1, sols_2;

	// returns the best parameter choice for (r,N,d) for given (m,w,lam), where lam is the proportional list size.
	static void optimal_params_blockwise(const uint64_t n, const uint64_t w, const double lam,
					    uint64_t *r, uint64_t *N, uint64_t *d) {
		*r = (lam*n/log2(n));
		double Hi = H1(1. - (((*r)-1)*lam)/double(*r));
		double w_star = 2*Hi*(1-Hi);
		// *d = n * H1(1. - (double((*r)-1.0)*lam/double(*r)));
		if (w > w_star)
			*d = 1./2. * (1. - sqrt(1. - 2.*w));
		else
			*d = Hi;

		double q = calc_q(n, double(w)/double(n), double(*d)/double(n));
		*N = uint64_t (n/q);
	}

	// perform an stupid slow quadratic in List size NN search. just for debugging.
	int NN() {
		// Reset everything.
		sols_1.resize(0);
		sols_2.resize(0);
		// perform a quadratic Search.
		return quadratic_NN(0, L1.size(), 0, L2.size());
	}

	// perform a stupid NN search so long you find that stupid golden solution.
	int NN(const NNContainer &gold1, const NNContainer &gold2) {
		sols_1.resize(0);
		sols_2.resize(0);

		bool not_found = true;
		int i;
		while (not_found)  {
			quadratic_NN(0, L1.size(), 0, L2.size(), &gold1, &gold2);
			ASSERT(sols_1.size() == sols_2.size());

			for (i = solution_searched; i < sols_1.size(); ++i) {
				if (gold1.is_equal(sols_1[i]) && gold2.is_equal(sols_2[i])){
					not_found = false;
					break;
				}
			}
			solution_searched = sols_1.size();

		}

		// we dont want, that this function is optimized out in synthetic benchmarks
		return i;
	}

	// internal quadratic NN function.
	// returns -1 if nothing found or find_all==true
	// else returns the position of the golden solution.
	int quadratic_NN(const uint64_t start1, const uint64_t end1,
	                 const uint64_t start2, const uint64_t end2,
	                 const NNContainer *gold1 = nullptr, const NNContainer *gold2 = nullptr) {
		ASSERT(start1 < end1 && start2 < end2 && end1 <= L1.size() && end2 <= L2. size());
		NNContainer tmp;

		int found = -1;

		for (int i = start1; i < end1; ++i) {
			const auto a = L1[i];
			for (int j = start2; j < end2; ++j) {
				const auto b = L2[j];

				// check if weight is omega
				auto w_ =  NNContainer::add(tmp, a, b);
				if (w_ == w){
					found += 1;

					if ((gold1 != nullptr) && (gold2 != nullptr)) {
						if (gold1->is_equal(a) && gold2->is_equal(b)){
							sols_1.push_back(a);
							sols_2.push_back(b);
							return found;
						}

						continue;
					}

					if (!find_all) {
						sols_1.push_back(a);
						sols_2.push_back(b);
						return found;
					} else {
						if (!already_found(a, b)) {
							sols_1.push_back(a);
							sols_2.push_back(b);
						}
					}
				}
			}
		}

		// a little hacky. but this forces all algorithms to run completely
		if (find_all)
			return -1;
		else
			return found;
	}

	/// creates two random lists L1, L2
	/// pos will be position of the golden nn element with distance w
	// If 'windowed' is set to 'true' this function ensures that the weight of every element is equally split
	// over 'k' blocks each of length 'G_n/k'
	static void create_test_lists(NNList &L1, NNList &L2, const uint64_t size, const uint64_t w,
	                              uint64_t &pos1, uint64_t &pos2, const bool windowed=false, const uint64_t k = 0) {
		L1.clear();
		L2.clear();
		L1.resize(size);
		L2.resize(size);


		// choose random positions
		pos1 = (NNContainer::random_limb()) % size;
		pos2 = (NNContainer::random_limb()) % size;

		// choose random lists.
		for (int i = 0; i < size; ++i) {
			L1[i].random();
			L2[i].random();
		}

		// set the `golden shower` nn elements
		NNContainer tmp;
		if (windowed && (k != 0))
			tmp.random_with_weight_per_windows(w/k, k);
		else
			tmp.random_with_weight(w);

		//std::cout << "weight element: " << tmp << "\n";
		NNContainer::add(L1[pos1], tmp, L2[pos2]);
	}

	// creates two lists of size 'size', where each element has weight 'g'. Afterwards the golden solution with weight
	// difference 'w' is implanted at random positions, which are returned in (&pos1, &pos2).
	// If 'windowed' is set to 'true' this function ensures that the weight of every element is equally split
	// over 'k' blocks each of length 'G_n/k'
	static void create_test_lists_with_distribution(NNList &L1, NNList &L2, const uint64_t size, const uint64_t g, const uint64_t w,
	                              uint64_t &pos1, uint64_t &pos2, const bool windowed=false, const uint64_t k = 0) {
		L1.clear();
		L2.clear();
		L1.resize(size);
		L2.resize(size);

		// choose random positions
		pos1 = (NNContainer::random_limb()) % size;
		pos2 = (NNContainer::random_limb()) % size;

		// choose random lists.
		for (int i = 0; i < size; ++i) {
			L1[i].random_with_weight(g);
			L2[i].random_with_weight(g);
		}

		// set the `golden` nn elements
		NNContainer tmp;
		if (windowed && (k != 0))
			tmp.random_with_weight_per_windows(w/k, k);
		else
			tmp.random_with_weight(w);

		NNContainer::add(L1[pos1], tmp, L2[pos2]);
	}

	// after successful execution call this via
	//      nn.print_result(gold1, gold2);
	// returns true if every element which is found is correct AND the golden solution was also found.
	// else returns false;
	bool print_result(const NNContainer &gold1, const NNContainer &gold2){
		information();
		ASSERT(sols_1.size() == sols_2.size());

		bool ret = true;

		if ((!sols_1.empty())) {
			std::cout << "Number of Solutions:" << sols_1.size() << "\n";

			// first check if the golden solution was found.
			int golden_found = -1;
			for (int i = 0; i < sols_1.size(); ++i) {
				if (gold1.is_equal(sols_1[i]) && gold2.is_equal(sols_2[i])){
					golden_found = i;
					break;
				}
			}

			if (golden_found != -1) {
#if defined(SOLUTION_LOGGING)
				std::cout << "golden solution found at pos: " << golden_found << "\n";
				std::cout << sols_1[golden_found] << " w:" << sols_1[golden_found].weight() << "\n";
				std::cout << sols_2[golden_found] << " w:" << sols_2[golden_found].weight() << "\n\n";
#endif
			} else {
#if defined(SOLUTION_LOGGING)
				std::cout << "golden solution NOT found: " << golden_found << "\n\n";
#endif
				ret = false;
			}

			for (int i = 0; i < sols_1.size(); ++i) {
#if defined(SOLUTION_LOGGING)

				const auto sol1 = sols_1[i];
				const auto sol2 = sols_2[i];

				auto i1 = std::find_if(L1.begin(), L1.end(),
				                       [&sol1](const NNContainer &e1) {;
					                       return sol1.is_equal(e1);
				                       }
				);
				auto i2 = std::find_if(L2.begin(), L2.end(),
				                       [&sol2](const NNContainer &e2) {;
					                       return sol2.is_equal(e2);
				                       }
				);

				if (i1 == L1.end()) { std::cout << "nn.sol_1 NOT found\n"; }
				if (i2 == L2.end()) { std::cout << "nn.sol_2 NOT found\n"; }

				std::cout << "Solution Found:\n";
				std::cout << sol1 << " w:" << sol1.weight() << "\n";
				std::cout << sol2 << " w:" << sol2.weight() << "\n";
				std::cout << "current pos of sol1: " << i1 - L1.begin() << "\n";
				std::cout << "current pos of sol2: " << i2 - L2.begin() << "\n\n";
#endif
				NNContainer tmp;
				const uint64_t w_ = NNContainer::add(tmp, sols_1[i], sols_2[i]);
				if (w_ != w)
					ret = false;

				if (tmp.weight() != w)
					ret = false;
			}
		} else {
			std::cout << "Nothing Found\n";
			ret = false;
		}

		return ret;
	}

};

#endif //NN_CODE_WINDOWED_NN_H

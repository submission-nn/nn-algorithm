#ifndef NN_CODE_INDYK_MOTWANI_H
#define NN_CODE_INDYK_MOTWANI_H

#include <sys/mman.h>
#include <algorithm>

#include "helper.h"
#include "container.h"
#include "list.h"
#include "nn.h"

class IndykMontwani : public NearestNeighbor {
private:
	const uint64_t l;

#ifdef PERFOMANCE_LOGGING
	const uint64_t max_window_size = 0;
#endif

public:
	IndykMontwani(NNList &L1, NNList &L2, const uint64_t w, const uint64_t l, const bool find_all=true) : l(l) ,
	                NearestNeighbor(L1, L2, w, 0, 0, 0, find_all, 0){
		ASSERT(L1.size() == L2.size());
	}

	int indyk_NN(const NNContainer &gold1, const NNContainer &gold2) {
		bool not_found = true;
		uint64_t n_solutions = 0;
		NNContainer z, tmp;

		while (not_found) {
			z.random_with_weight(l);
			std::cout << "z:" << z << "\n" << std::flush;

			std::sort(L1.begin(), L1.end(),
				[z](const NNContainer &e1, const NNContainer &e2) {
					return e1.is_lower(e2, z);
				}
			);

			std::sort(L2.begin(), L2.end(),
				[z](const NNContainer &e1, const NNContainer &e2) {
					return e1.is_lower(e2, z);
				}
			);

			uint64_t i = 0, j = 0;
			while (i < L1.size() && j < L2.size()){
				if (L1[i].is_lower(L2[j], z))
					i += 1;
				else if (L2[j].is_lower(L1[i], z))
					j += 1;
				else {
					// equal
					uint64_t i_max, j_max;
					for (i_max = i + 1; i_max < L1.size() && L1[i].is_equal(L1[i_max], z); i_max++) {}
					for (j_max = j + 1; j_max < L2.size() && L2[j].is_equal(L2[j_max], z); j_max++) {}

					// Sanity Check
					ASSERT(i_max - i < L1.size());
#ifdef PERFOMANCE_LOGGING
					if ((i_max - i) > max_window_size)
						max_window_size = i_max - i;
#endif
					uint64_t j_prev = j;
					for (; i < i_max; i++) {
						for (j = j_prev; j < j_max; ++j) {
							// now the final test.
							if (NNContainer::add(tmp, L1[i], L2[j]) == w){
								sols_1.push_back(L1[i]);
								sols_2.push_back(L2[j]);

								if (gold1.is_equal(sols_1[n_solutions]) && gold2.is_equal(sols_2[n_solutions])){
									return n_solutions;
								}

								n_solutions += 1;
								// not_found = false;
							}
						}
					}
				}
			}
		}

		return n_solutions;
	}
};

#endif //NN_CODE_INDYK_MOTWANI_H

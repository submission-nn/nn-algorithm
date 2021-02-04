#ifndef SMALLSECRETLWE_LIST_H
#define SMALLSECRETLWE_LIST_H

// stl include
#include <vector>           // main data container

using NNList      = std::vector<NNContainer>;

///
/// \tparam Element
/// \param out
/// \param obj
/// \return
std::ostream &operator<<(std::ostream &out, const NNList &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << obj[i] << "\n";
	}

	return out;
}

#endif //SMALLSECRETLWE_LIST_H

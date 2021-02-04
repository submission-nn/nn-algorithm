#ifndef NN_CODE_CONTAINER_H
#define NN_CODE_CONTAINER_H

#include <cstdint>
#include <sys/random.h>
#include "helper.h"

static uint64_t xorshf96() {          //period 2^96-1
	static uint64_t x=123456789u, y=362436069u, z=521288629u;

	uint64_t t;
	x ^= x << 16u;
	x ^= x >> 5u;
	x ^= x << 1u;

	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

/* n = size of buffer in bytes, */
static int fastrandombytes(void *buf, size_t n){
	uint64_t *a = (uint64_t *)buf;

	const uint32_t rest = n%8;
	const size_t limit = n/8;
	size_t i = 0;

	for (; i < limit; ++i) {
		a[i] = xorshf96();
	}

	// last limb
	uint8_t *b = (uint8_t *)buf;
	b += n - rest;
	uint64_t limb = xorshf96();
	for (size_t j = 0; j < rest; ++j) {
		b[j] = (limb >> (j*8u)) & 0xFFu;
	}

	return 0;
}

/* n = bytes. */
inline static int fastrandombytes_uint64_array(uint64_t *buf, size_t n){
	void *a = (void *)buf;
	fastrandombytes(a, n);
	return 0;
}

static uint64_t fastrandombytes_uint64() {
#define UINT64_POOL_SIZE 256    // page should be 512 * 8 Byte
	static uint64_t tmp[UINT64_POOL_SIZE];
	static size_t counter = 0;

	if (counter == 0){
		fastrandombytes_uint64_array(tmp, UINT64_POOL_SIZE * 8 );
		counter = UINT64_POOL_SIZE;
	}

	counter -= 1;
	return tmp[counter];
}


template<unsigned int length >
class BinaryContainer {
private:
	using BINARY_TYPE = uint64_t;

	constexpr static uint64_t popcount(BINARY_TYPE x) { return __builtin_popcountll(x); }
	constexpr static uint64_t popcount(BINARY_TYPE x, uint64_t mask) { return __builtin_popcountll(x & mask); }

public:
	// how many limbs to we need and how wide are they.
	constexpr static uint16_t limb_bits_width() { return limb_bytes_width() * 8; };
	constexpr static uint16_t limb_bytes_width() { return sizeof(BINARY_TYPE); };

	// round a given amount of 'in' bits to the nearest limb excluding the the lowest overflowing bits
	// eg 13 -> 64
	constexpr static uint16_t round_up(uint16_t in) { return round_up_to_limb(in) * limb_bits_width(); }
	constexpr static uint16_t round_up_to_limb(uint16_t in) {return (in/limb_bits_width())+1; }

	// the same as above only rounding down
	// 13 -> 0
	constexpr static uint16_t round_down(uint16_t in) { return round_down_to_limb(in) * limb_bits_width(); }
	constexpr static uint16_t round_down_to_limb(uint16_t in) { return (in/limb_bits_width()); }

	constexpr static uint16_t compute_limbs() { return (length+limb_bits_width()-1)/limb_bits_width(); };

	// calculate from a bit position 'i' the mask to set it.
	constexpr static BINARY_TYPE mask(uint16_t i ) {
		ASSERT(i <= length && "wrong access index");
		return (BINARY_TYPE(1) << (BINARY_TYPE(i)%limb_bits_width()));
	}

	// given the i-th bit this function will return a bits mask where the lower 'i' bits are set. Everything will be
	// realigned to limb_bits_width().
	constexpr static BINARY_TYPE lower_mask(uint16_t i ) {
		ASSERT(i <= length && "wrong access index");
		BINARY_TYPE u = i%limb_bits_width();
		BINARY_TYPE c = ((BINARY_TYPE(1) << u) - 1);

		return c;
	}

	// given the i-th bit this function will return a bits mask where the higher (n-i)bits are set.
	constexpr static BINARY_TYPE higher_mask(uint16_t i ) {
		ASSERT(i <= length && "wrong access index");
		return (~lower_mask(i));
	}

	// given the i-th bit this function will return a bits mask where the lower 'n-i' bits are set. Everything will be
	// realigned to limb_bits_width().
	constexpr static BINARY_TYPE lower_mask_inverse(uint16_t i ) {
		ASSERT(i <= length && "wrong access index");
		BINARY_TYPE u = BINARY_TYPE(i)%limb_bits_width();
		if (u == 0) { return -1; }
		return (BINARY_TYPE(1) << (limb_bits_width()-u)) - BINARY_TYPE(1);
	}

	// given the i-th bit this function will return a bits mask where the higher (i) bits are set.
	constexpr static BINARY_TYPE higher_mask_inverse(uint16_t i ) {
		ASSERT(i <= length && "wrong access index");
		return (~lower_mask_inverse(i));
	}

	// not shifted.
	constexpr BINARY_TYPE get_bit(uint16_t i) const { return __data[round_down_to_limb(i)] & mask(i); }
	// shifted.
	constexpr bool get_bit_shifted(uint16_t i) const { return (__data[round_down_to_limb(i)] & mask(i)) >> i; }

	void write_bit(const uint16_t pos, const uint8_t bit) { __data[pos/limb_bits_width()] = (__data[pos/limb_bits_width()] & ~(BINARY_TYPE(1u) << (pos%limb_bits_width()))) | (BINARY_TYPE(bit) << (pos%limb_bits_width())); }

	// actual data container.
	std::array<BINARY_TYPE, compute_limbs()> __data;
	constexpr static uint64_t upper = compute_limbs();
	constexpr static uint64_t apply_mask = length%limb_bits_width()==0 ? lower_mask(length)-1 : lower_mask(length);

	// hack it like
	class reference {
		friend class BinaryContainer;

		// pointer to the limb
		BINARY_TYPE     *wp;
		// bit position in the whole data array.
		const size_t 	 mask_pos;

		// left undefined
		reference();

	public:
		reference(BinaryContainer &b, size_t pos) : mask_pos(mask(pos)){
			wp = &b.data().data()[round_down_to_limb(pos)];
		}

		reference(const reference&) = default;

		~reference() = default;

		// For b[i] = __x;
		reference& operator=(bool x) {
			if (x)
				*wp |= mask_pos;
			else
				*wp &= ~mask_pos;
			return *this;
		}

		// For b[i] = b[__j];
		reference& operator=(const reference& j) {
			if (*(j.wp) & j.mask_pos)
				*wp |= mask_pos;
			else
				*wp &= ~mask_pos;
			return *this;
		}

		// Flips the bit
		bool operator~() const { return (*(wp) & mask_pos) == 0; }

		// For __x = b[i];
		operator bool() const {
			return (*(wp) & mask_pos) != 0;
		}

		// For b[i].flip();
		reference& flip() {
			*wp ^= mask_pos;
			return *this;
		}

		unsigned int get_data() {
			return bool();
		}
	};
	friend class reference;


	/// default constructor
	BinaryContainer(): __data() { ASSERT(length > 0 && "length __MUST__ > 0"); }

	/// Copy Constructor
	BinaryContainer(const BinaryContainer& a) : __data(a.__data) {}

	/// zero the complete data vector
	constexpr void zero() {
		LOOP_UNROLL();
		for (unsigned int i = 0; i < compute_limbs(); ++i) {
			__data[i] = 0;
		}
	}

	// seth the whole array to 'fff...fff'
	void one() {
		for (int i = 0; i < compute_limbs()-1; ++i) {
			__data[i] = BINARY_TYPE(-1);
		}
		__data[compute_limbs()-1] =  BINARY_TYPE(-1) & apply_mask;
	}

	BINARY_TYPE static random_limb() { BINARY_TYPE t; getrandom(&t, limb_bytes_width(), 0); return t; }

	/// set the whole data array on random data.
	void random() {
		if constexpr (length < 64) {
			__data[0] = fastrandombytes_uint64() & apply_mask;
		} else {
			fastrandombytes_uint64_array(__data.data(), compute_limbs() * limb_bytes_width());
		}
	}

	// this function actually aplly a random permutations on the container data to simulate a random choice of bits.
	void random_with_weight(const uint64_t w){
		zero();

		for (int i = 0; i < w; ++i) {
			write_bit(i, true);
		}

		// now permute
		for (int i = 0; i < length; ++i) {
			uint64_t pos = random_limb() % (length - i);
			auto t = get_bit_shifted(i);
			write_bit(i, get_bit_shifted(i+pos));
			write_bit(i+pos, t);
		}
	}

	// this function will split the container into 'k' buckets and set the weight 'w' in  each window
	void random_with_weight_per_windows(const uint64_t w, const uint64_t k) {

		std::vector<uint64_t> buckets_windows{};

		// this stipid approach needs to be done, because if w is not dividing n the last bits would be unused.
		buckets_windows.resize(k+1);
		for (int i = 0; i < k; ++i) {
			buckets_windows[i] = i*length/k;
		}
		buckets_windows[k] = length;

		// clear everything.
		zero();

		for (int i = 0; i < k; ++i) {
			uint64_t cur_offset = buckets_windows[i];
			uint64_t windows_length = buckets_windows[i+1] - buckets_windows[i];

			for (int j = 0; j < w; ++j) {
				write_bit(cur_offset + j, true);
			}

			// now permute
			for (int l = 0; l < windows_length; ++l) {
				uint64_t pos = random_limb() % (windows_length - l);
				auto t = get_bit_shifted(cur_offset + l);
				write_bit(cur_offset + l, get_bit_shifted(cur_offset+l+pos));
				write_bit(cur_offset+l+pos, t);
			}
		}
	}

	// swap the two bits i, j
	void swap(const uint16_t i, const uint16_t j) {
		ASSERT(i < length && j < length);
		auto t = get_bit_shifted(i);
		write_bit(i, get_bit_shifted(j));
		write_bit(j, t);
	}


	inline static int add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                       const uint64_t k_lower, const uint64_t k_upper) {
		ASSERT(k_upper <= length && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		const BINARY_TYPE lmask = higher_mask(k_lower%64);
		const BINARY_TYPE rmask = k_upper%64 == 0 ? BINARY_TYPE(-1) : lower_mask(k_upper%64);
		const int64_t lower_limb = k_lower/64;
		const int64_t higher_limb = (k_upper-1)/64;

		if (lower_limb == higher_limb) {
			const BINARY_TYPE mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			BINARY_TYPE tmp1 = (v3.__data[lower_limb] & ~(mask));
			BINARY_TYPE tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			auto b = popcount(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb+1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcount(v3.__data[i]);
		}

		BINARY_TYPE tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		BINARY_TYPE tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		BINARY_TYPE tmp11 = (v3.__data[lower_limb] & ~(lmask));
		BINARY_TYPE tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1^tmp11;
		v3.__data[higher_limb]= tmp2^tmp21;

		cnorm += popcount(tmp1);
		cnorm += popcount(tmp2);

		return cnorm;
	}

	inline static int add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) {
		int r = 0;

		LOOP_UNROLL();
		for (int64_t i = 0; i < upper-1; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			r += __builtin_popcountll(v3.__data[i]);
		}

		v3.__data[upper-1] = (v1.__data[upper-1] ^ v2.__data[upper-1]) & apply_mask;
		return r + __builtin_popcountll(v3.__data[upper-1]);
	}

	inline int add(BinaryContainer const &v){
		return add(*this, *this, v);
	}

	inline static int add_simple(BinaryContainer const &v1, BinaryContainer const &v2, const uint64_t limb, const uint64_t mask) {
		ASSERT(limb < compute_limbs());
		auto b = v1.__data[limb];
		b ^= v2.__data[limb];
		b &= mask;

		return __builtin_popcountll(b);
	}

	inline static int add_simple2(BinaryContainer const &v1, BinaryContainer const &v2, BinaryContainer const &v3, const uint64_t limb, const uint64_t mask) {
		ASSERT(limb < compute_limbs());
		auto b = v2.__data[limb];
		b ^=v3.__data[limb];
		b &= mask;
		return __builtin_popcountll(b);
	}

	inline static void and_(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) {
		int r = 0;

		LOOP_UNROLL();
		for (int64_t i = 0; i < upper-1; ++i) {
			v3.__data[i] = v1.__data[i] & v2.__data[i];
		}

		v3.__data[upper-1] = (v1.__data[upper-1] & v2.__data[upper-1]) & apply_mask;
	}


	inline static bool cmp(BinaryContainer const &v1, BinaryContainer const &v2) {
		auto masked_compare = [](const BINARY_TYPE limb1, const BINARY_TYPE limb2, const BINARY_TYPE mask) {
			return ((limb1&mask) == (limb2&mask));
		};

		auto compare = [](const BINARY_TYPE limb1, const BINARY_TYPE limb2) {
			return (limb1 == limb2);
		};

		// the two offsets lay in two different limbs
		// first check the highest limb with the mask
		if (!masked_compare(v1.__data[upper-1], v2.__data[upper-1], apply_mask))
			return false;

		// check all limbs in the middle
		for(uint64_t i = 0; i < upper -1; i++) {
			if (!compare(v1.__data[i], v2.__data[i]))
				return false;
		}

		return true;
	}


	inline bool is_equal(const BinaryContainer &obj) const { return cmp(*this, obj); }

	// special compare
	inline bool is_equal(const BinaryContainer &obj, const BinaryContainer &z) const {
		if constexpr (compute_limbs() == 1) {
			return (__data[0]&z.__data[0] & apply_mask) == (obj.__data[0]&z.__data[0] & apply_mask);
		}

		// the two offsets lay in two different limbs
		// first check the highest limb with the mask
		if ((__data[upper-1]&z.__data[upper-1] & apply_mask) != (obj.__data[upper-1]&z.__data[upper-1] & apply_mask))
			return false;

		// check all limbs in the middle
		for(uint64_t i = 0; i < upper-1; i++) {
			if ((__data[i]&z.__data[i]) != (obj.__data[i]&z.__data[i]))
				return false;
		}

		return true;
	}


	// Special compare
	inline bool is_lower(BinaryContainer const &obj, BinaryContainer const &z) const {
		if constexpr (compute_limbs() == 1) {
			return (__data[0]&z.__data[0]& apply_mask) < (obj.__data[0]&z.__data[0]& apply_mask);
		}

		auto b = upper;

		// the two offsets lay in two different limbs
		// first check the highest limb with the mask
		if ((__data[upper-1]&z.__data[upper-1] & apply_mask) < (obj.__data[upper-1]&z.__data[upper-1] & apply_mask))
			return true;

		// check all limbs in the middle
		for(uint64_t i = upper-1; i > 0 ; i--) {
			if ((__data[i-1]&z.__data[i-1]) < (obj.__data[i-1]&z.__data[i-1]))
				return true;
		}

		return false;
	}

	inline bool is_lower(BinaryContainer const &obj) const {
		if constexpr (compute_limbs() == 1) {
			return (__data[0] & apply_mask) < (obj.__data[0]& apply_mask);
		}

		// the two offsets lay in two different limbs
		// first check the highest limb with the mask
		if ((__data[upper-1] & apply_mask) < (obj.__data[upper-1] & apply_mask))
			return true;

		// check all limbs in the middle
		for(uint64_t i = upper-1; i > 0 ; i--) {
			if (__data[i-1] < obj.__data[i-1])
				return true;
		}

		return false;
	}

	inline uint64_t weight() const {

		uint64_t r = 0;
		for (int i = 0; i < upper-1; ++i) {
			r += __builtin_popcountll(__data[i]);
		}
		return r + __builtin_popcountll(__data[upper-1]&apply_mask);
	}

	inline uint64_t weight(const uint64_t k_lower, const uint64_t k_upper){
		ASSERT(k_upper <= length && "ERROR static void sub not correct k_upper");
		ASSERT(k_lower < k_upper && "ERROR static void sub not correct k_lower");

		uint64_t lower = round_down_to_limb(k_lower);
		uint64_t upper = round_down_to_limb(k_upper);

		uint64_t l_mask = higher_mask(k_lower);
		uint64_t u_mask = lower_mask(k_upper);

		//if only one limb to check
		if(lower == upper){
			uint64_t b = (l_mask & u_mask);
			uint64_t c = uint64_t(__data[lower]);
			uint64_t d = uint64_t(b) & uint64_t(c);
			uint64_t w_ = __builtin_popcountll(d);
			//std::cout << "w" << w << "\n";
			return w_;
		}

		//if at least two limbs
		uint64_t weight = __builtin_popcountll(l_mask&__data[lower]);
		weight += __builtin_popcountll(u_mask&__data[upper]);
		for(uint64_t i=lower+1;i<upper;++i)
			weight+=__builtin_popcountll(__data[i]);

		return weight;
	}

	reference operator[](size_t pos) { return reference(*this, pos); }
	constexpr bool operator[](const size_t pos) const { return (__data[round_down_to_limb(pos)] & mask(pos)) != 0; }

	BinaryContainer& operator =(BinaryContainer const &obj) {
		if (this != &obj) { // self-assignment check expected
			std::copy(&obj.__data[0], &obj.__data[0] + obj.__data.size(), &this->__data[0]);
		}

		return *this;
	}

	BinaryContainer& operator =(BinaryContainer &&obj) noexcept {
		if (this != &obj) { // self-assignment check expected really?
			__data = std::move(obj.__data);
		}

		return *this;
	}

	auto& data() { return __data; };
	const auto& data() const { return __data; };

	//BINARY_TYPE data(uint64_t index) { ASSERT(index < length); return get_bit_shifted(index); }
	const bool data(uint64_t index) const { ASSERT(index < length); return get_bit_shifted(index); }

	BINARY_TYPE get_type() {return __data[0]; }

	// Q: this is kinda funny. Why is this legal C++?
	// A: Why not?
	inline constexpr uint64_t size() const { return length; }
	inline constexpr static uint64_t get_size() { return length; }
	inline constexpr uint64_t limbs() const { return compute_limbs(); }
};

template<unsigned int length >
std::ostream& operator<< (std::ostream &out, const BinaryContainer<length> &obj) {
#ifdef COMPRESS_OUTPUT
	if constexpr (length > 10) {
		constexpr uint64_t offset = 5;
		for (uint64_t i = 0; i < offset; ++i) {
			out << obj[i] << " ";
		}
		out << "... ";

		for (uint64_t i = obj.size()-offset; i < obj.size(); ++i) {
			out << obj[i] << " ";
		}
	}else {
		for (uint64_t i = 0; i < obj.size(); ++i) {
			out << obj[i] << " ";
		}
	}
#else
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << obj[i] << " ";
	}

	// std::cout << " w: " << obj.weight() << "\n";
	// std::cout << "\n";
#endif

	return out;
}

template<unsigned int length >
void print(const BinaryContainer<length> &obj, const uint64_t k_lower=0, const uint64_t k_higher=length) {
	ASSERT(k_lower < k_higher && k_higher <= length);

#ifdef COMPRESS_OUTPUT
#error not implemented
#else
	for (uint64_t i = 0; i < obj.size(); ++i) {
		if (i == k_lower)
			std::cout << " |";
		if (i == k_higher)
			std::cout << "|  ";
		std::cout << obj[i];
		if (i != k_higher-1)
			std::cout << " ";
	}
#endif
}


template<unsigned int length >
void print(const BinaryContainer<length> &obj, const std::vector<uint64_t> &buckets) {
#ifdef COMPRESS_OUTPUT
#error not implemented
#else

	uint64_t counter = 0;

	for (uint64_t i = 0; i < obj.size(); ++i) {
		if (i == buckets[counter++])
			std::cout << " |";
		if (i == buckets[counter++])
			std::cout << "|  ";
		std::cout << obj[i];
		if (i !=  buckets[counter]-1)
			std::cout << " ";
	}
#endif
}



#ifndef G_n
#define G_n 100
#endif
using NNContainer = BinaryContainer<G_n>;


#endif //NN_CODE_CONTAINER_H

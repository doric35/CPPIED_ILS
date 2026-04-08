#include "../../include/heuristics/luby.hpp"

int luby::operator()(int i) {
    while (true) {
        int k = static_cast<int>(std::floor(std::log2(i))) + 1;
        int k_pow = 1 << k;              // 2^k
        int k_minus_pow = k_pow >> 1;    // 2^{k-1}

        if (i == k_pow - 1)
            return scale * k_minus_pow;

        i = i - k_minus_pow + 1;
    }
}
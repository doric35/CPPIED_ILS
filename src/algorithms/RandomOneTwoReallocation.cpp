#include "../../include/algorithms/RandomOneTwoReallocation.hpp"

void RandomOneTwoReallocation::setIncumbentMove() {
    if (!candidates.empty()) {
        std::vector<cost_t> gains;
        gains.reserve(candidates.size());
        for (auto &c: candidates)
            gains.push_back(c.gain);

        int i = ris_heuristic::softmaxSample(gains, rng);
        incumbent = candidates[i];
        correctIncumbentIndexInsertion();
    }
}

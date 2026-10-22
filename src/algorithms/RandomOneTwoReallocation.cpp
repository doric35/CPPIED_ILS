#include "../../include/algorithms/RandomOneTwoReallocation.hpp"

void RandomOneTwoReallocation::setIncumbentMove() {
    if (!candidates.empty()) {
        std::vector<cost_t> gains;
        gains.reserve(candidates.size());
        for (auto &c: candidates)
            gains.push_back(c.gain);
        applyNegativeGainsCorrection(gains);
        int i = ris_heuristic::softmaxSample(gains, rng);
        incumbent = candidates[i];
        correctIncumbentIndexInsertion();
    }
}

void RandomOneTwoReallocation::applyNegativeGainsCorrection(std::vector<cost_t> &gains) {
    cost_t g_min = std::ranges::min(gains);
    if (g_min < cost_t{0,0})
        for (auto& g : gains) g -= g_min;
}

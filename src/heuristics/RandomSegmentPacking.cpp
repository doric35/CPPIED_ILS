#include "../../include/heuristics/RandomSegmentPacking.hpp"

std::vector<int> RandomSegmentPacking::solve() {
    packing.items.clear();
    packing.mask = Eigen::VectorXi(instance.req.size());
    packing.mask.setZero();
    std::shuffle(candidates.begin(), candidates.end(), rng);
    packShuffledCandidates();
    return packing.items;
}

void RandomSegmentPacking::packShuffledCandidates() {
    for (int i : candidates){
        if (!coverage.intersect_mask(solution, solution.path[i], packing.mask)){
            packing.items.push_back(i);
            coverage.mask(solution, solution.path[i], packing.mask);
        }
    }
}

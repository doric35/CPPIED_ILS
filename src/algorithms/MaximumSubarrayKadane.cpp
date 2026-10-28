#include "../../include/algorithms/MaximumSubarrayKadane.hpp"

SubArray MaximumSubarrayKadane::solve() {
    if (!costs.empty()) {
        current   = {0, 0, costs[0]};
        incumbent = current;
        for (int i = 1; i < costs.size(); ++i) {
            current.end = i;
            current.value += costs[i];
            current   = std::max(current, SubArray{i, i, costs[i]});
            incumbent = std::max(current, incumbent);
        }
    }
    return incumbent;
}
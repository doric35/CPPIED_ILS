#include "../../include/neighborhoods/neighborhood_reduce.hpp"

void neighborhood_reduce::splitMoves(std::vector<int> &candidates) {
    splitOddEvenMoves(candidates);
    splitHorizontalVerticalMoves(candidates);
}

void neighborhood_reduce::applyMoves(std::vector<int> &selectedMoves) {
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    removeSelectedMoves(selectedMoves);
    solution.cost = geometry.cost(solution);
}

void neighborhood_reduce::removeSelectedMoves(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves) {
        coverage.remove(solution, solution.path[i]);
        solution.path[i] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    }

    std::erase_if(solution.path, [&](const segment& s){ return !path_engine::is_node(s.source);});
}
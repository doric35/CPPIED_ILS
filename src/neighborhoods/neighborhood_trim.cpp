#include "../../include/neighborhoods/neighborhood_trim.hpp"

void neighborhood_trim::splitMoves(std::vector<int> &candidates) {
    splitOddEvenMoves(candidates);
    splitHorizontalVerticalMoves(candidates);
}

void neighborhood_trim::applyMoves(std::vector<int> &selectedMoves) {
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    swapSelectedMoves(selectedMoves);
    std::erase_if(solution.path, [](const segment& s){ return !path_engine::is_node(s.source); });
}

void neighborhood_trim::swapSelectedMoves(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves)
        swap(i);
}

void neighborhood_trim::swap(int i) {
    coverage.remove(solution, solution.path[i]);
    if (path_engine::is_node(candidatesMoves[i].s.source))
        coverage.insert(solution, candidatesMoves[i].s);
    solution.path[i] = candidatesMoves[i].s;
    solution.cost -= candidatesMoves[i].gain;
}

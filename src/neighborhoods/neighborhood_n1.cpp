#include "../../include/neighborhoods/neighborhood_n1.hpp"

void neighborhood_n1::splitMoves(std::vector<int> &candidates) {
    splitOddEvenMoves(candidates);
    splitHorizontalVerticalMoves(candidates);
}

void neighborhood_n1::applyMoves(std::vector<int> &selectedMoves) {
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    swapSelectedMoves(selectedMoves);
}

void neighborhood_n1::swapSelectedMoves(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves)
        swap(i);
    std::erase_if(solution.path, [&](const segment& s){ return !path_engine::is_node(s.source);});
}

void neighborhood_n1::swap(int i) {
    coverage.remove(solution, solution.path[i]);
    coverage.insert(solution, candidatesMoves[i].s);
    solution.path[i] = candidatesMoves[i].s;
    solution.cost -= candidatesMoves[i].gain;
}

#include "../../include/neighborhoods/neighborhood_n12.hpp"

void neighborhood_n12::splitMoves(std::vector<int> &candidates) {
    std::vector<Key>     packing_keys = constructPackingKeys(candidates);
    RandomKeyPacking     heuristic(packing_keys);
    std::vector<Packing> packing = heuristic.solve();
    splitPackings(candidates, packing);
    splitHorizontalVerticalMoves(candidates);
}

std::vector<Key> neighborhood_n12::constructPackingKeys(std::vector<int> &candidates) {
    std::vector<Key> packing_keys;
    packing_keys.reserve(candidates.size());
    for (int i : candidates){
        Key key{i,candidatesMoves[i].reallocationIndexes[0],candidatesMoves[i].reallocationIndexes[1]};
        key.l = key.l < 0 ? key.l = -2 : key.l;
        key.k = key.k < 0 ? key.k = -2 : key.k;
        packing_keys.push_back(key);
    }
    return packing_keys;
}

void neighborhood_n12::splitPackings(std::vector<int> &candidates, std::vector<Packing> &packing) {
    candidates.clear(); splitIndices.clear();
    for (auto & j : packing){
        for (int candidate : j)
            candidates.push_back(candidate);
        splitIndices.push_back(static_cast<int>(candidates.size()));
    }
    if (!splitIndices.empty())
        splitIndices.pop_back();
}

void neighborhood_n12::applyMoves(std::vector<int> &selectedMoves) {
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    removeSelectedMovesCoverage(selectedMoves);
    addSelectedMovesCoverage(selectedMoves);
    computeMarginalGain(selectedMoves);
    setOrderedMoves(selectedMoves);
    swapSelectedMoves(selectedMoves);
}

void neighborhood_n12::removeSelectedMovesCoverage(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves)
        coverage.remove(solution, solution.path[i]);
}

void neighborhood_n12::setOrderedMoves(std::vector<int> &selectedMoves) {
    orderedMoves.clear();
    orderedMoves.reserve(2*selectedMoves.size());
    for (int i : selectedMoves){
        if (path_engine::is_node(candidatesMoves[i].replacements[0].source))
            orderedMoves.emplace_back(candidatesMoves[i].replacements[0], candidatesMoves[i].reallocationIndexes[0]);
        if (path_engine::is_node(candidatesMoves[i].replacements[1].source))
            orderedMoves.emplace_back(candidatesMoves[i].replacements[1], candidatesMoves[i].reallocationIndexes[1]);
    }
    auto cmpPositions = [](const std::pair<segment, int>& a, const std::pair<segment, int>& b){
        return a.second < b.second;
    };
    std::stable_sort(orderedMoves.begin(), orderedMoves.end(), cmpPositions);
}

void neighborhood_n12::addSelectedMovesCoverage(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves){
        const auto& move = candidatesMoves[i];
        if (path_engine::is_node(move.replacements[0].source))
            coverage.insert(solution, move.replacements[0]);
        if (path_engine::is_node(move.replacements[1].source))
            coverage.insert(solution, move.replacements[1]);
    }
}

void neighborhood_n12::computeMarginalGain(std::vector<int> &selectedMoves) {
    for (int i : selectedMoves)
        solution.cost -= candidatesMoves[i].gain;
}

void neighborhood_n12::swapSelectedMoves(std::vector<int> &selectedMoves) {
    std::vector<segment> moved_path;
    moved_path.reserve(solution.path.size());
    auto selected_it = selectedMoves.begin();
    auto moves_it = orderedMoves.begin();
    for (int i=0; i< solution.path.size(); i++){
        if (moves_it != orderedMoves.end() && moves_it->second == i){
            moved_path.push_back(moves_it->first);
            moves_it++;
        }
        if (moves_it != orderedMoves.end() && moves_it->second == i){
            moved_path.push_back(moves_it->first);
            moves_it++;
        }
        if (selected_it == selectedMoves.end() || i != *selected_it){
            moved_path.push_back(solution.path[i]);
        } else selected_it++;
    }
    for (; moves_it != orderedMoves.end(); moves_it++)
        moved_path.push_back(moves_it->first);
    solution.path.swap(moved_path);
}


#include "../../include/neighborhoods/neighborhood_n21.hpp"

void neighborhood_n21::splitMoves(std::vector<int> &candidates) {
    std::vector<Key>     packing_keys = constructPackingKeys(candidates);
    RandomKeyPacking     heuristic(packing_keys);
    std::vector<Packing> packing = heuristic.solve();
    splitPackings(candidates, packing);
    splitHorizontalVerticalMoves(candidates);
}

std::vector<Key> neighborhood_n21::constructPackingKeys(std::vector<int> &candidates) {
    std::vector<Key> packing_keys;
    packing_keys.reserve(candidates.size());
    for (int i : candidates){
        Key key{i,candidatesMoves[i].removalIndex,candidatesMoves[i].reallocationIndex};
        key.l = key.l < 0 ? key.l = -2 : key.l;
        key.k = key.k < 0 ? key.k = -2 : key.k;
        packing_keys.push_back(key);
    }
    return packing_keys;
}

void neighborhood_n21::splitPackings(std::vector<int> &candidates, std::vector<Packing> &packing) {
    candidates.clear();
    splitIndices.clear();
    for (auto & j : packing){
        for (int candidate : j)
            candidates.push_back(candidate);
        splitIndices.push_back(static_cast<int>(candidates.size()));
    }
    if (!splitIndices.empty())
        splitIndices.pop_back();
}

void neighborhood_n21::applyMoves(std::vector<int> &moves) {
    selectedMoves = moves;
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    removeSelectedMovesCoverage();
    addSelectedMovesCoverage();
    computeMarginalGain();
    setOrderedMoves();
    swapSelectedMoves();
}

void neighborhood_n21::removeSelectedMovesCoverage() {
    for (int i : selectedMoves) {
        coverage.remove(solution, solution.path[i]);
        if (candidatesMoves[i].removalIndex >=0 )
            coverage.remove(solution, solution.path[candidatesMoves[i].removalIndex]);
    }
}

void neighborhood_n21::setOrderedMoves() {
    orderedMoves.clear();
    orderedMoves.reserve(selectedMoves.size());
    for (int i : selectedMoves){
        if (path_engine::is_node(candidatesMoves[i].replacement.source))
            orderedMoves.emplace_back(candidatesMoves[i].replacement, candidatesMoves[i].reallocationIndex);
        if (candidatesMoves[i].removalIndex >=0 )
            pairedRemovals.push_back(candidatesMoves[i].removalIndex);
    }
    auto cmpPositions = [](const std::pair<segment, int>& a, const std::pair<segment, int>& b){
        return a.second < b.second;
    };
    std::stable_sort(orderedMoves.begin(), orderedMoves.end(), cmpPositions);
    std::sort(pairedRemovals.begin(), pairedRemovals.end());
}

void neighborhood_n21::addSelectedMovesCoverage() {
    for (int i : selectedMoves) {
        const auto &move = candidatesMoves[i];
        if (path_engine::is_node(move.replacement.source))
            coverage.insert(solution, move.replacement);
    }
}

void neighborhood_n21::computeMarginalGain() {
    for (int i : selectedMoves)
        solution.cost -= candidatesMoves[i].gain;
}

void neighborhood_n21::swapSelectedMoves() {
    std::vector<segment> moved_path;
    moved_path.reserve(solution.path.size());
    solution.path.swap(moved_path);
    auto selected_it = selectedMoves.begin();
    auto removed_it = pairedRemovals.begin();
    auto moves_it = orderedMoves.begin();
    for (int i=0; i< moved_path.size(); i++){
        if (isMoveToInsert(i, moves_it)){
            solution.path.push_back(moves_it->first);
            moves_it++;
        }
        if (!isSelectedOrRemoved(i, selected_it, removed_it))
            solution.path.push_back(moved_path[i]);
        else if (isSelected(i, selected_it))
            selected_it++;
        else
            removed_it++;
    }
    pushRemainingMoves(moves_it);
}

bool neighborhood_n21::isSelectedOrRemoved(int index, std::vector<int>::const_iterator selector_iterator,
                                           std::vector<int>::const_iterator removal_iterator) {
    return isSelected(index, selector_iterator) || isRemoved(index, removal_iterator);
}

bool neighborhood_n21::isSelected(int index, std::vector<int>::const_iterator selector_iterator) {
    return selector_iterator != selectedMoves.cend() && index == *selector_iterator;
}

bool neighborhood_n21::isRemoved(int index, std::vector<int>::const_iterator removal_iterator) {
    return removal_iterator != pairedRemovals.cend() && index == *removal_iterator;
}

bool neighborhood_n21::isMoveToInsert(int index, std::vector<std::pair<segment, int>>::const_iterator move_iterator) {
    return move_iterator != orderedMoves.cend() && move_iterator->second == index;
}

void neighborhood_n21::pushRemainingMoves(std::vector<std::pair<segment, int>>::iterator move_iterator) {
    for (; move_iterator != orderedMoves.end(); move_iterator++)
        solution.path.push_back(move_iterator->first);
}

Interval neighborhood_n21::transformToInterval(int i) {
    Interval interval = SimpleNeighborhood::transformToInterval(i);
    interval = correctIntervalEncoding(interval);
    return interval;
}

Interval neighborhood_n21::correctIntervalEncoding(cppied_intervals::Interval &interval) {
    if (candidatesMoves[interval.id].removalIndex >= 0) {
        Interval other = getSecondInterval(interval, candidatesMoves[interval.id].removalIndex);
        interval = getIntervalsUnion(interval, other);
    }
    return interval;
}

Interval neighborhood_n21::getSecondInterval(cppied_intervals::Interval &first, int second) {
    Interval interval{};
    if (geometry.is_horizontal(solution.path[first.id]) == geometry.is_horizontal(solution.path[second]))
        interval = SimpleNeighborhood::transformToInterval(second);
    else
        interval = getSecondPerpendicularInterval(first, second);
    return interval;
}

Interval neighborhood_n21::getSecondPerpendicularInterval(cppied_intervals::Interval &first, int second) {
    Interval interval{};
    int reference_point;
    if (first.rotated)
        reference_point = geometry.get_row(solution.path[second]);
    else
        reference_point = geometry.get_column(solution.path[second]);
    interval.x1 = reference_point - problem.max_range;
    interval.x2 = reference_point + problem.max_range - 1;
    return interval;
}

Interval neighborhood_n21::getIntervalsUnion(cppied_intervals::Interval &first, cppied_intervals::Interval &second) {
    Interval interval_union = first;
    interval_union.x1 = std::min(first.x1, second.x1);
    interval_union.x2 = std::max(first.x2, second.x2);
    return interval_union;
}

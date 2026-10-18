#include "../../include/algorithms/MaximumTwoOneReallocation.hpp"

TwoOneReallocationMove MaximumTwoOneReallocation::solve() {
    if (coverage.can_remove(solution, solution.path[index])){
        incumbent = {{path_engine::NULL_NODE, path_engine::NULL_NODE},
                     -1, -1, geometry.removal_gain(solution, solution.path.begin() + index)};
    } else {
        coverage.remove(solution, solution.path[index]);
        solution.path.erase(solution.path.begin() + index);
        setRemovalCandidates();
        setCandidatesGains();
        setIncumbentMove();
        coverage.insert(solution, firstRemoved);
        solution.path.insert(solution.path.begin() + index, firstRemoved);
        correctIncumbentIndexes();
    }
    return incumbent;
}

void MaximumTwoOneReallocation::setRemovalCandidates() {
    setCandidateBox();
    for (int i=1; i<solution.path.size(); i++){
        if (isValidCandidate(solution.path[i]))
            candidates.emplace_back(segment{path_engine::NULL_NODE, path_engine::NULL_NODE},
                                    -1, i, cost_t{MAX_INT, MAX_INT});
    }
}

void MaximumTwoOneReallocation::setCandidateBox() {
    if (geometry.is_horizontal(firstRemoved))
        setCandidateBoxFromHorizontalFirst();
    else
        setCandidateBoxFromVerticalFirst();
}

bool MaximumTwoOneReallocation::isValidCandidate(segment& s) {
    return geometry.is_in_rectangle(s, box);
}

void MaximumTwoOneReallocation::setCandidateBoxFromHorizontalFirst() {
    segment non_oriented_removed = firstRemoved;
    geometry.correct_segment_direction(non_oriented_removed, direction::E);
    box.ul = {0, std::max(0, geometry.get_row(non_oriented_removed) - 2*instance.max_range)};
    box.lr = {geometry.n_cols, std::min(geometry.n_rows, geometry.get_row(non_oriented_removed) + 2*instance.max_range)};
}

void MaximumTwoOneReallocation::setCandidateBoxFromVerticalFirst() {
    segment non_oriented_removed = firstRemoved;
    geometry.correct_segment_direction(non_oriented_removed, direction::S);
    box.ul = {std::max(0, geometry.get_column(non_oriented_removed) - 2*instance.max_range),0};
    box.lr = {std::min(geometry.n_cols, geometry.get_column(non_oriented_removed) + 2*instance.max_range),
                      geometry.n_cols};
}

cost_t MaximumTwoOneReallocation::getGain(TwoOneReallocationMove &move) {
    segment move_segment_copy = solution.path[move.removalIndex];
    cost_t removal_gain = geometry.removal_gain(solution, solution.path.begin() + move.removalIndex);
    coverage.remove(solution, solution.path[move.removalIndex]);
    solution.path.erase(solution.path.begin() + move.removalIndex);
    MinimumSegmentAllocation allocation_algorithm(instance, coverage, geometry, solution);
    Allocation allocation = allocation_algorithm.solve();
    coverage.insert(solution, move_segment_copy);
    solution.path.insert(solution.path.begin() + move.removalIndex, move_segment_copy);
    move.gain = path_engine::is_node(allocation.s.source) ? removalGain + removal_gain - allocation.cost :
            cost_t{MIN_INT, MIN_INT};
    move.replacement = allocation.s;
    move.reallocationIndex = allocation.index <= move.removalIndex ? allocation.index : allocation.index + 1;
    return move.gain;
}

void MaximumTwoOneReallocation::correctIncumbentIndexes() {
    incumbent.removalIndex = incumbent.removalIndex < index ? incumbent.removalIndex : incumbent.removalIndex + 1;
    incumbent.reallocationIndex = incumbent.reallocationIndex <= index ? incumbent.reallocationIndex : incumbent.reallocationIndex + 1;
}

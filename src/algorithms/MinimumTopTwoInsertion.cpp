#include "../../include/algorithms/MinimumTopTwoInsertion.hpp"

TopTwoInsertion MinimumTopTwoInsertion::solve() {
    solveNonTerminalInsertions();
    solveTerminalInsertions();
    return incumbent;
}

void MinimumTopTwoInsertion::solveNonTerminalInsertions() {
    segment flipped = geometry.flip_segment(nonOrientedSegment);
    auto insertionCmp = [](
            const SegmentInsertion& a, const SegmentInsertion& b){
        return a.insertion_cost < b.insertion_cost;
    };
    for (int i=1; i< solution.path.size(); ++i){
        SegmentInsertion insertion = {nonOrientedSegment, i,
                                      geometry.insert_between_gain(solution.path[i-1], nonOrientedSegment, solution.path[i])};
        SegmentInsertion flippedInsertion = {flipped, i,
                                             geometry.insert_between_gain(solution.path[i-1], flipped, solution.path[i])};
        SegmentInsertion argmin = std::min(insertion, flippedInsertion, insertionCmp);
        updateIncumbent(argmin);
    }
}

void MinimumTopTwoInsertion::updateIncumbent(SegmentInsertion& move) {
    if (move.insertion_cost < incumbent.bestInsertion.insertion_cost){
        std::swap(incumbent.bestInsertion, incumbent.backupInsertion);
        std::swap(move, incumbent.bestInsertion);
    } else if (move.insertion_cost < incumbent.backupInsertion.insertion_cost)
        std::swap(move, incumbent.backupInsertion);
}

void MinimumTopTwoInsertion::solveTerminalInsertions() {
    auto insertionCmp = [](
            const SegmentInsertion& a, const SegmentInsertion& b){
        return a.insertion_cost < b.insertion_cost;
    };
    segment flipped = geometry.flip_segment(nonOrientedSegment);
    const int i = static_cast<int>(solution.path.size());
    SegmentInsertion insertion = {nonOrientedSegment, i,
                                  geometry.insert_tail_gain(solution.path[i-1], nonOrientedSegment)};
    SegmentInsertion flippedInsertion = {flipped, i,
                                         geometry.insert_tail_gain(solution.path[i-1], flipped)};
    SegmentInsertion argmin = std::min(insertion, flippedInsertion, insertionCmp);
    updateIncumbent(argmin);
}
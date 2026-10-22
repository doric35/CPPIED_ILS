#include "../../include/algorithms/MinimumSegmentInsertion.hpp"

SegmentInsertion MinimumSegmentInsertion::solve() {
    if (solution.path.size() > 1)
        solveNonTerminalInsertions();
    if (!solution.path.empty())
        solveTerminalInsertions();
    return incumbent;
}

void MinimumSegmentInsertion::solveNonTerminalInsertions() {
    auto indexes_view = std::views::iota(1, static_cast<int>(solution.path.size()));
    segment flipped = geometry.flip_segment(nonOrientedSegment);
    auto insertion_cost = [&](const int i){
        return geometry.insert_between_gain(solution.path[i-1], nonOrientedSegment, solution.path[i]);};
    auto insertion_cost_flipped = [&](const int i){
        return geometry.insert_between_gain(solution.path[i-1], flipped, solution.path[i]);};
    auto argmin = std::ranges::min_element(indexes_view, std::ranges::less{}, insertion_cost);
    auto argmin_flipped = std::ranges::min_element(indexes_view, std::ranges::less{}, insertion_cost_flipped);
    const cost_t cost_s = geometry.insert_between_gain(solution.path[*argmin-1], nonOrientedSegment, solution.path[*argmin]);
    const cost_t cost_s_flipped = geometry.insert_between_gain(solution.path[*argmin_flipped-1], flipped, solution.path[*argmin_flipped]);
    if (cost_s <= cost_s_flipped)
        incumbent = {nonOrientedSegment, *argmin, cost_s};
    else
        incumbent = {flipped, *argmin_flipped, cost_s_flipped};
}

void MinimumSegmentInsertion::solveTerminalInsertions() {
    segment flipped = geometry.flip_segment(nonOrientedSegment);
    const cost_t cost_s = geometry.insert_tail_gain(solution.path.back(), nonOrientedSegment);
    const cost_t cost_s_flipped = geometry.insert_tail_gain(solution.path.back(), flipped);
    int position = static_cast<int>(solution.path.size());
    if (cost_s <= incumbent.insertion_cost)
        incumbent = {nonOrientedSegment, position, cost_s};
    if (cost_s_flipped <= incumbent.insertion_cost)
        incumbent = {flipped, position, cost_s_flipped};
}
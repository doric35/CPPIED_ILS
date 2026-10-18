#include "../../include/algorithms/MaximumSegmentTrim.hpp"

ReplacementMove MaximumSegmentTrim::solve() {
    segment trimmed = trimSegmentOvercoveredHeadAndTail(solution.path[index]);
    incumbent = {trimmed, cost_t{MIN_INT, MIN_INT}};
    if (path_engine::is_node(trimmed.source))
        incumbent.gain = getGain(incumbent);
    else
        incumbent.gain = geometry.removal_gain(solution, solution.path.begin() + index);
    return incumbent;
}

cost_t MaximumSegmentTrim::getGain(ReplacementMove &move) {
    auto current_segment = std::next(solution.path.begin(), index);
    cost_t gain = geometry.removal_gain(solution, current_segment);
    if (current_segment != std::prev(solution.path.end()))
        gain -= geometry.insert_between_gain(
                *std::prev(current_segment),move.s,*std::next(current_segment));
    else
        gain -= geometry.insert_tail_gain(*std::prev(current_segment), move.s);
    return gain;
}

bool MaximumSegmentTrim::isValidMove(ReplacementMove &move) {
    return true;
}
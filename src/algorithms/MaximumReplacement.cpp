#include "../../include/algorithms/MaximumReplacement.hpp"

ReplacementMove MaximumReplacement::solve() {
    coverage.remove(solution, solution.path[index]);
    setUnsatisfiedCells();
    if (!unsatisfiedCells.empty()) {
        setUnsatisfiedRectangle();
        setCompleteCoverageSegments();
        flipCandidates();
        setCandidatesGains();
        setPositiveCandidates();
        setIncumbentMove();
    } else{
        incumbent.s = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        incumbent.gain = geometry.removal_gain(solution, solution.path.begin() + index);
    }
    coverage.insert(solution, solution.path[index]);
    return incumbent;
}

void MaximumReplacement::setUnsatisfiedCells() {
    coverage.unsatisfied_cells(solution, solution.path[index], unsatisfiedCells);
}

void MaximumReplacement::setUnsatisfiedRectangle() {
    coverage.unsatisfied_rectangle(unsatisfiedCells, unsatisfiedBox);
}

void MaximumReplacement::setCompleteCoverageSegments() {
    setCompleteCoverageHorizontalSegments();
    setCompleteCoverageVerticalSegments();
}

void MaximumReplacement::setCompleteCoverageHorizontalSegments() {
    int row = std::max(0, unsatisfiedBox.lr.y - instance.max_range);
    int bound = std::min(geometry.n_rows, unsatisfiedBox.ul.y + instance.max_range);

    //Horizontal segments
    for (; row <= bound; ++row){
        segment s{row * geometry.n_cols + unsatisfiedBox.ul.x,row * geometry.n_cols + unsatisfiedBox.lr.x - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        candidates.emplace_back(s, cost_t{MIN_INT, MIN_INT});
    }
}

void MaximumReplacement::setCompleteCoverageVerticalSegments() {
    int col = std::max(0, unsatisfiedBox.lr.x - instance.max_range);
    int bound = std::min(geometry.n_cols, unsatisfiedBox.ul.x + instance.max_range);

    for (; col <= bound; ++col){
        segment s{geometry.horizontal_bound + (col * geometry.n_rows) + unsatisfiedBox.ul.y,
                  geometry.horizontal_bound + col * geometry.n_rows + unsatisfiedBox.lr.y - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        candidates.emplace_back(s, cost_t{MIN_INT, MIN_INT});
    }
}

void MaximumReplacement::flipCandidates() {
    const int n = candidates.size();
    for (int i=0; i< n; ++i){
        ReplacementMove flipped = candidates[i];
        flipped.s = geometry.flip_segment(flipped.s);
        candidates.push_back(flipped);
    }
}

bool MaximumReplacement::isValidMove(ReplacementMove &move) {
    return coverage.insertion_satisfies(solution, move.s, unsatisfiedCells);
}

cost_t MaximumReplacement::getGain(ReplacementMove &move) {
    auto current_segment = std::next(solution.path.begin(), index);
    cost_t gain = geometry.removal_gain(solution, current_segment);
    if (current_segment != std::prev(solution.path.end())) {
        gain -= geometry.insert_between_gain(
                *std::prev(current_segment),move.s,*std::next(current_segment));
    } else {
        gain -= geometry.insert_tail_gain(*std::prev(current_segment), move.s);
    }
    return gain;
}
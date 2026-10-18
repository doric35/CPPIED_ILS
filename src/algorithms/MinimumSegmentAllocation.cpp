#include "../../include/algorithms/MinimumSegmentAllocation.hpp"

Allocation MinimumSegmentAllocation::solve() {
    setUnsatisfiedCells();
    setUnsatisfiedRectangle();
    setCompleteCoverageSegments();
    filterCandidates();
    for (auto& s : candidates)
        updateIncumbent(s);

    return incumbent;
}

void MinimumSegmentAllocation::setUnsatisfiedCells() {
    coverage.unsatisfied_cells(solution, unsatisfiedCells);
}

void MinimumSegmentAllocation::setUnsatisfiedRectangle() {
    coverage.unsatisfied_rectangle(unsatisfiedCells, unsatisfiedBox);
}

void MinimumSegmentAllocation::setCompleteCoverageSegments() {
    setCompleteCoverageSegmentsHorizontal();
    setCompleteCoverageSegmentsVertical();
}

void MinimumSegmentAllocation::filterCandidates(){
    std::erase_if(candidates, [&](const segment& s){ return !isValidAllocation(s); });
}

void MinimumSegmentAllocation::setCompleteCoverageSegmentsHorizontal() {
    int row = std::max(0, unsatisfiedBox.lr.y - instance.max_range);
    int bound = std::min(geometry.n_rows, unsatisfiedBox.ul.y + instance.max_range);

    //Horizontal segments
    for (; row <= bound; ++row){
        segment s{row * geometry.n_cols + unsatisfiedBox.ul.x,row * geometry.n_cols + unsatisfiedBox.lr.x - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        candidates.push_back(s);
    }
}

void MinimumSegmentAllocation::setCompleteCoverageSegmentsVertical() {
    int col = std::max(0, unsatisfiedBox.lr.x - instance.max_range);
    int bound = std::min(geometry.n_cols, unsatisfiedBox.ul.x + instance.max_range);

    for (; col <= bound; ++col){
        segment s{geometry.horizontal_bound + (col * geometry.n_rows) + unsatisfiedBox.ul.y,
                  geometry.horizontal_bound + col * geometry.n_rows + unsatisfiedBox.lr.y - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        candidates.push_back(s);
    }
}

bool MinimumSegmentAllocation::isValidAllocation(const segment& s) {
    return coverage.insertion_satisfies(solution, s, unsatisfiedCells);
}

void MinimumSegmentAllocation::updateIncumbent(segment &s) {
    auto top_insertion = geometry.get_top_k_insertion(solution, s, 1);
    auto top_insertion_reversed = geometry.get_top_k_insertion(solution, geometry.flip_segment(s), 1);
    Allocation trial{s, top_insertion.front().first, top_insertion.front().second};
    if (top_insertion_reversed.front().second < trial.cost)
        trial = {geometry.flip_segment(s), top_insertion_reversed.front().first, top_insertion_reversed.front().second};

    if (trial.cost < incumbent.cost)
        incumbent = trial;
}
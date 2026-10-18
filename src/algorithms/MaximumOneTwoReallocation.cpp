#include "../../include/algorithms/MaximumOneTwoReallocation.hpp"
#include "../../include/algorithms/MaximumSegmentTrim.hpp"

OneTwoReallocationMove MaximumOneTwoReallocation::solve() {
    coverage.remove(solution, solution.path[index]);
    std::vector<int> unsatisfied_cells = getUnsatisfiedCells();
    if (!unsatisfied_cells.empty()) {
        setCandidates(unsatisfied_cells);
        filterCandidates(unsatisfied_cells);
        trimCandidates();
        setCandidatesTopInsertions();
        setIncumbentMove();
    } else
        setSimpleRemovalIncumbent();
    coverage.insert(solution, solution.path[index]);
    return incumbent;
}

std::vector<int> MaximumOneTwoReallocation::getUnsatisfiedCells() {
    std::vector<int> cells;
    coverage.unsatisfied_cells(solution, solution.path[index], cells);
    return cells;
}

void MaximumOneTwoReallocation::setCandidates(std::vector<int>& unsatisfiedCells) {
    std::vector<segment> first_level_candidates = getReplacements(unsatisfiedCells);
    for (auto& s : first_level_candidates)
        setSecondLevelCandidates(unsatisfiedCells, s);
}

std::vector<segment> MaximumOneTwoReallocation::getReplacements(std::vector<int> &unsatisfiedCells) {
    iRectangle box = getUnsatisfiedRectangle(unsatisfiedCells);
    return getPartialCoverageSegments(box);
}

std::vector<segment> MaximumOneTwoReallocation::getPartialCoverageSegments(iRectangle& box) {
    std::vector<segment> segments = getPartialCoverageHorizontalSegments(box);
    std::vector<segment> verticals = getPartialCoverageVerticalSegments(box);
    segments.insert(segments.end(), verticals.begin(), verticals.end());
    return segments;
}

std::vector<segment> MaximumOneTwoReallocation::getPartialCoverageHorizontalSegments(iRectangle& box) {
    std::vector<segment> segments;
    int row = std::max(0, box.ul.y - (instance.max_range-1));
    int geometry_bound = std::min(geometry.n_rows, box.lr.y + (instance.max_range-1));

    //Horizontal segments
    for (; row <= geometry_bound; ++row){
        segment s{row * geometry.n_cols + box.ul.x,row * geometry.n_cols + box.lr.x - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        segments.push_back(s);
    }
    return segments;
}

std::vector<segment> MaximumOneTwoReallocation::getPartialCoverageVerticalSegments(iRectangle& box) {
    std::vector<segment> segments;
    int col = std::max(0, box.ul.x - (instance.max_range - 1));
    int geometry_bound = std::min(geometry.n_cols, box.lr.x + (instance.max_range-1));

    for (; col <= geometry_bound; ++col){
        segment s{geometry.horizontal_bound + (col * geometry.n_rows) + box.ul.y,
                  geometry.horizontal_bound + col * geometry.n_rows + box.lr.y - 1};
        if (s.source == s.target) s.target = path_engine::NULL_NODE;
        segments.push_back(s);
    }
    return segments;
}

void MaximumOneTwoReallocation::setSecondLevelCandidates(std::vector<int> unsatisfiedCells, segment &firstLevel) {
    coverage.insert(solution, firstLevel);
    coverage.filter_unsatisfied_cells(solution, unsatisfiedCells);
    if (unsatisfiedCells.empty()){
        std::array<segment, 2> replacements = {
                firstLevel,
                {path_engine::NULL_NODE, path_engine::NULL_NODE}
        };
        std::array<int, 2> reallocationIndexes = {-1, -1};
        cost_t gain = {MIN_INT, MIN_INT};
        candidates.emplace_back(replacements, reallocationIndexes, gain);
    } else {
        std::vector<segment> second_level_segments = getReplacements(unsatisfiedCells);
        for (auto &s: second_level_segments) {
            std::array<segment, 2> replacements        = {firstLevel, s};
            std::array<int, 2>     reallocationIndexes = {-1, -1};
            cost_t                 gain                = {MIN_INT, MIN_INT};
            candidates.emplace_back(replacements, reallocationIndexes, gain);
        }
    }
    coverage.remove(solution, firstLevel);
}

void MaximumOneTwoReallocation::filterCandidates(std::vector<int>& unsatisfiedCells) {
    auto unsatisfied_coverage = [&](const OneTwoReallocationMove& move){
        bool sat;
        if (path_engine::is_node(move.replacements[1].source))
            sat = !coverage.insertion_satisfies(
                    solution, {move.replacements[0], move.replacements[1]}, unsatisfiedCells);
        else
            sat = !coverage.insertion_satisfies(solution, move.replacements[0], unsatisfiedCells);
        return sat;
    };
    std::erase_if(candidates, unsatisfied_coverage);
}

void MaximumOneTwoReallocation::trimCandidates() {
    std::vector<OneTwoReallocationMove> trimmed;
    for (auto& move : candidates){
        if (geometry.is_node(move.replacements[1].source))
            move.replacements[0] = trimSegmentOvercoveredHeadAndTail(move.replacements[0]);

        if (!processedPairs.contains(SegmentPair(move.replacements[0], move.replacements[1]))) {
            trimmed.push_back(move);
            processedPairs.insert(SegmentPair(move.replacements[0], move.replacements[1]));
        }
    }
    candidates.swap(trimmed);
}

void MaximumOneTwoReallocation::setCandidatesTopInsertions() {
    segment cp = solution.path[index];
    solution.path.erase(solution.path.begin() + index);
    for (auto& move : candidates)
        setMoveTopInsertion(move);
    solution.path.insert(solution.path.begin() + index, cp);
}

void MaximumOneTwoReallocation::setMoveTopInsertion(OneTwoReallocationMove &move) {
    MinimumSegmentPairInsertion insertion_algorithm(
            instance, coverage, geometry,
            solution, move.replacements[0],
            move.replacements[1], insertionCostBound);
    try {
        Insertion insertion = insertion_algorithm.solve();
        move.replacements = {insertion.segments.first, insertion.segments.second};
        move.gain = removalGain - insertion.cost;
        move.reallocationIndexes = {insertion.indexes.first, insertion.indexes.second};
    } catch (DominatedInsertion& e){
        move.gain = {MIN_INT,MIN_INT};
    }
}

segment MaximumOneTwoReallocation::trimSegment(segment &s) {
    return trimSegmentOvercoveredHeadAndTail(s);
}

void MaximumOneTwoReallocation::setSimpleRemovalIncumbent() {
    incumbent.replacements[0] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    incumbent.replacements[1] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    incumbent.reallocationIndexes[0] = -1;
    incumbent.reallocationIndexes[1] = -1;
    incumbent.gain = geometry.removal_gain(solution, solution.path.begin() + index);
}

iRectangle MaximumOneTwoReallocation::getUnsatisfiedRectangle(std::vector<int>& unsatisfiedCells) {
    iRectangle box;
    coverage.unsatisfied_rectangle(unsatisfiedCells, box);
    return box;
}

void MaximumOneTwoReallocation::setIncumbentMove() {
    if (!candidates.empty()) {
        auto cmpGain = [](const OneTwoReallocationMove &a, const OneTwoReallocationMove &b) {
            return a.gain < b.gain;
        };
        incumbent = *std::max_element(candidates.begin(), candidates.end(), cmpGain);
        correctIncumbentIndexInsertion();
    }
}

void MaximumOneTwoReallocation::correctIncumbentIndexInsertion(){
    incumbent.reallocationIndexes[0] += static_cast<int>(incumbent.reallocationIndexes[0] >= index+1);
    incumbent.reallocationIndexes[1] += static_cast<int>(incumbent.reallocationIndexes[1] >= index+1);
}
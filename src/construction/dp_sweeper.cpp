#include "../../include/construction/dp_sweeper.hpp"

void DpSweeper::construct(const std::function<void(cppied_solution&)>& f) {
    mPenalty = 0.01;
    setRootSegmentIfEmptySolution();
    setUnsatisfiedCells();
    while (!unsatisfiedCells.empty())
        addCoverageSegmentSet(f);
    solveTspReduction();
}

void DpSweeper::setRootSegmentIfEmptySolution() {
    if (solution.path.empty()) {
        segment initial = {problem.initial_position, path_engine::NULL_NODE};
        coverage.insert(solution, initial);
        solution.path.push_back(initial);
    }
}

void DpSweeper::setUnsatisfiedCells() {
    unsatisfiedCells.clear();
    coverage.unsatisfied_cells(solution, unsatisfiedCells);
}

void DpSweeper::addCoverageSegmentSet(const std::function<void(cppied_solution&)>& save_history) {
    std::array<sweeper::SegmentSet, 2> candidates = constructCandidateSets();
    filterCandidateSetsWithSpacingConstraint(candidates);
    selectCandidateSet(candidates);
    setUnsatisfiedCells();
    save_history(solution);
}

std::array<sweeper::SegmentSet, 2> DpSweeper::constructCandidateSets() {
    iRectangle box;
    coverage.unsatisfied_rectangle(unsatisfiedCells, box);
    std::array<sweeper::SegmentSet, 2> S{{
        {{}, {}},
        {{}, {}}
    }};
    auto set_candidates = [&](const segment &seg) {
        auto [s, g] = maximumSubarray(seg);
        int orientation = geometry.is_horizontal(seg) ? 0 : 1;
        S[orientation].segments.push_back(s);
        S[orientation].gains.push_back(g);
    };
    for_each_segment(box, set_candidates);
    return S;
}

void DpSweeper::filterCandidateSetsWithSpacingConstraint(std::array<sweeper::SegmentSet, 2> &candidateSets) {
    filterCandidateSet(candidateSets[0]);
    filterCandidateSet(candidateSets[1]);
}

void DpSweeper::filterCandidateSet(sweeper::SegmentSet &set) {
    MaximumSpaceConstrainedArraySelection selection_algorithm(2*problem.max_range, set.gains);
    auto selection = selection_algorithm.solve();
    sweeper::SegmentSet filtered;
    for (int i : selection) {
        filtered.segments.push_back(set.segments[i]);
        filtered.gains.push_back(set.gains[i]);
    }
    std::swap(filtered, set);
}

void DpSweeper::selectCandidateSet(std::array<sweeper::SegmentSet, 2>& candidateSets) {
    int choice = mChoice(candidateSets);
    if (!isValidChoice(choice))
        choice = int(!horizontalDominatesVertical(candidateSets));

    if (candidateSets[choice].segments.empty())
        mPenalty /= 10.0;

    addSelectedSegmentsToSolution(candidateSets[choice]);
}

bool DpSweeper::horizontalDominatesVertical(std::array<sweeper::SegmentSet, 2> &candidateSets) {
    sweeper::LexicographicSegmentSetValue firstValues = getLexicographicValues(candidateSets[0]);
    sweeper::LexicographicSegmentSetValue secondValues = getLexicographicValues(candidateSets[1]);
    return firstValues >= secondValues;
}

sweeper::LexicographicSegmentSetValue DpSweeper::getLexicographicValues(sweeper::SegmentSet &set) {
    sweeper::LexicographicSegmentSetValue values{};
    values.gain = std::ranges::fold_left(set.gains, 0.0, std::plus{});
    values.negative_size = -static_cast<int>(set.segments.size());
    values.negative_length = -std::accumulate(set.segments.begin(), set.segments.end(), 0,
                                              [&](int c, const segment& s){ return c + geometry.cost(s).length + 1;});
    return values;
}

bool DpSweeper::isValidChoice(int choice) {
    return choice == 0 || choice == 1;
}

void DpSweeper::addSelectedSegmentsToSolution(const sweeper::SegmentSet &segments) {
    for (auto& s : segments.segments){
        solution.path.push_back(s);
        coverage.insert(solution, s);
    }
}

void DpSweeper::setInitialSolutionAlternateOrientations() {
    for (int i = 1; i < solution.path.size(); i += 2)
        solution.path[i] = geometry.flip_segment(solution.path[i]);
}

void DpSweeper::solveTspReduction() {
    setInitialSolutionAlternateOrientations();
    solution.cost = geometry.cost(solution);
    neighborhood_tsp n(ctx, problem, solution);
    n.local_search();
}

std::pair<segment, double> DpSweeper::maximumSubarray(const segment &array) {
    std::vector<double> gains = constructArrayCosts(array);
    MaximumSubarrayKadane kadane_algorithm(gains);
    SubArray subarray = kadane_algorithm.solve();
    segment non_oriented = geometry.getNonOriented(array);
    segment subarray_segment = {non_oriented.source + subarray.start,
                                non_oriented.source + subarray.end};
    subarray_segment.target = subarray_segment.source != subarray_segment.target ? subarray_segment.target : path_engine::NULL_NODE;
    return {subarray_segment, subarray.value};
}

std::vector<double> DpSweeper::constructArrayCosts(const segment &array) {
    auto [u, v] = getArrayBounds(array);
    auto gainFunction = [&](int i){
        double vertex_gain = 0.0;
        for (SMdIt c(problem.s_pod, i); c; ++c){
            double local_gain = std::min(problem.req(c.index()) - solution.coverage(c.index()), c.value());
            local_gain = (local_gain >= 0) * local_gain - (local_gain < 0) * mPenalty;
            vertex_gain += local_gain;
        }
        return vertex_gain;
    };
    auto gains = std::views::iota(u, v+1)
            | std::views::transform(gainFunction)
            | std::views::transform([](auto x){ return x <= 0.0 ? MIN_INT : x; })
            | std::ranges::to<std::vector>();
    return gains;
}

std::pair<int,int> DpSweeper::getArrayBounds(const segment &array) {
    segment non_oriented = geometry.getNonOriented(array);
    int u = non_oriented.source;
    int v = std::max(array.source, array.target);
    return {u,v};
}

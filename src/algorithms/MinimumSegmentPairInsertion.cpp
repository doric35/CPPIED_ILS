#include "../../include/algorithms/MinimumSegmentPairInsertion.hpp"

Insertion MinimumSegmentPairInsertion::solve() {
    if (solution.path.size() > 1 && path_engine::is_node(secondSegment.source)) {
        solveDisjointInsertion();
        solvePairedInsertion();
    } else if (solution.path.size() > 1)
        solveUnitaryInsertion();
    else
        solveUnitPathEdgeCase();

    correctIncumbentSegmentsOrder();
    return incumbent;
}

void MinimumSegmentPairInsertion::solveUnitPathEdgeCase() {
    if (path_engine::is_node(secondSegment.source)) {
        for (auto &pair: pairedKernels) {
            cost_t cost = geometry.insert_tail_gain(solution.path.front(), pair.first);
            cost += geometry.insert_tail_gain(pair.first, pair.second);
            Insertion insertion{{1, 1}, cost, pair};
            updateIncumbent(insertion);
        }
    } else {
        cost_t cost = geometry.insert_tail_gain(solution.path.front(), firstSegment);
        Insertion insertion{{1, -1}, cost, {firstSegment, secondSegment}};
        updateIncumbent(insertion);
        cost = geometry.insert_tail_gain(solution.path.front(), firstReversedSegment);
        insertion = {{1, -1}, cost, {firstReversedSegment, secondSegment}};
        updateIncumbent(insertion);
    }
}

void MinimumSegmentPairInsertion::solveUnitaryInsertion() {
    std::array<std::pair<int, cost_t>, 4> first_picks = getSegmentTopInsertions(firstSegment);
    if (std::min(first_picks[0].second, first_picks[2].second) >= bound)
        throw DominatedInsertion("Segments can't be inserted at cost lower than the bound.");
    setIncumbentFromUnitaryInsertion(first_picks);
}

void MinimumSegmentPairInsertion::setIncumbentFromUnitaryInsertion(std::array<std::pair<int, cost_t>, 4> &pick) {
    Insertion trial;
    if (pick[0].second <= pick[2].second)
        trial = {{pick[0].first, -1}, pick[0].second, {firstSegment, {path_engine::NULL_NODE, path_engine::NULL_NODE}}};
    else
        trial = {{pick[2].first, -1}, pick[2].second, {firstReversedSegment, {path_engine::NULL_NODE, path_engine::NULL_NODE}}};
    updateIncumbent(trial);
}

void MinimumSegmentPairInsertion::solveDisjointInsertion() {
    std::array<std::pair<int, cost_t>, 4> first_picks = getSegmentTopInsertions(firstSegment);
    std::array<std::pair<int, cost_t>, 4> second_picks = getSegmentTopInsertions(secondSegment);
    if (picksAreDominated(first_picks, second_picks))
        throw DominatedInsertion("Segments can't be inserted at cost lower than the bound.");
    setIncumbentFromDisjointInsertions(first_picks, second_picks);
}

bool MinimumSegmentPairInsertion::picksAreDominated(std::array<std::pair<int, cost_t>, 4> &firstPick,
                                                    std::array<std::pair<int, cost_t>, 4> &secondPick) {
    cost_t first_best_pick_cost = std::min(firstPick[0].second, firstPick[2].second);
    cost_t second_best_pick_cost = std::min(secondPick[0].second, secondPick[2].second);
    return std::max(first_best_pick_cost, second_best_pick_cost) >= bound;
}

std::array<std::pair<int, cost_t>, 4> MinimumSegmentPairInsertion::getSegmentTopInsertions(segment &s) {
    std::array<std::pair<int, cost_t>, 4> top_k;

    std::uint64_t segment_encoding = encodeSegment(s.source, s.target);
    if (memoizedDisjointTopInsertions.contains(segment_encoding)){
        top_k = memoizedDisjointTopInsertions.at(segment_encoding);
    } else {
        segment reversed = geometry.flip_segment(s);
        std::vector<std::pair<int,cost_t>> top_2 = geometry.get_top_k_insertion(solution, s, 2);
        std::vector<std::pair<int,cost_t>> top_2_reversed = geometry.get_top_k_insertion(solution, reversed, 2);

        top_k[0] = top_2[0];
        top_k[1] = top_2[1];
        top_k[2] = top_2_reversed[0];
        top_k[3] = top_2_reversed[1];

        memoizedDisjointTopInsertions.insert({segment_encoding, top_k});
    }
    return top_k;
}

void MinimumSegmentPairInsertion::setIncumbentFromDisjointInsertions(std::array<std::pair<int, cost_t>, 4> &first_pick,
                                                                std::array<std::pair<int, cost_t>, 4> &second_pick) {
    std::array<Insertion,16> pairs = getDisjointInsertionPairs(first_pick, second_pick);
    setIncumbentFromDisjointInsertionPairs(pairs);
}

std::array<Insertion,16> MinimumSegmentPairInsertion::getDisjointInsertionPairs(
        std::array<std::pair<int, cost_t>, 4> &firstPick, std::array<std::pair<int, cost_t>, 4> &secondPick) {
    std::array<Insertion,16> paired_insertions;
    for (int i=0; i<4; i++){
        for (int j=0; j<4; j++){
            paired_insertions[(i*4) + j].indexes = {firstPick[i].first, secondPick[j].first};
            paired_insertions[(i*4) + j].cost = firstPick[i].second + secondPick[j].second;
            paired_insertions[(i*4) + j].segments.first = i >=2 ? firstReversedSegment : firstSegment;
            paired_insertions[(i*4) + j].segments.second = j >=2 ? secondReversedSegment : secondSegment;
        }
    }
    return paired_insertions;
}

void MinimumSegmentPairInsertion::setIncumbentFromDisjointInsertionPairs(std::array<Insertion, 16> insertionPairs) {
    auto cmpInsertions = [&](const Insertion& a, const Insertion& b){
        bool a_lower;
        const bool a_valid = a.indexes.first != a.indexes.second;
        const bool b_valid = b.indexes.first != b.indexes.second;
        if (a_valid == b_valid)
            a_lower = a.cost < b.cost;
        else
            a_lower = a_valid;
        return a_lower;
    };
    auto trial = std::min_element(insertionPairs.begin(), insertionPairs.end(), cmpInsertions);
    updateIncumbent(*trial);
}

void MinimumSegmentPairInsertion::solvePairedInsertion() {
    for (int i=1; i<= solution.path.size(); i++)
        pairedInsertionKernel(i);
}

void MinimumSegmentPairInsertion::pairedInsertionKernel(int index) {
    for (auto& pair : pairedKernels){
        Insertion trial{{index, index}, cost_t{0,0}, pair};
        trial.cost = getInsertionCost(trial);
        updateIncumbent(trial);
    }
}

void MinimumSegmentPairInsertion::updateIncumbent(Insertion& trial) {
    if (trial.cost < incumbent.cost)
        incumbent = trial;
    SegmentPair key(firstSegment, secondSegment);
    if (memoizedMinimalInsertions.contains(key)){
        if (memoizedMinimalInsertions.at(key).cost > trial.cost)
            memoizedMinimalInsertions[key] = trial;
    } else
        memoizedMinimalInsertions.insert({key,trial});
}

cost_t MinimumSegmentPairInsertion::getInsertionCost(segment &s, int index) {
    cost_t cost;
    if (index < solution.path.size())
        cost = geometry.insert_between_gain(
                *std::next(solution.path.begin(), index-1),
                s,
                *std::next(solution.path.begin(), index));
    else
        cost = geometry.insert_tail_gain(*std::prev(solution.path.end()), s);
    return cost;
}

cost_t MinimumSegmentPairInsertion::getInsertionCost(Insertion &insertion) {
    cost_t cost;
    if (insertion.indexes.first < solution.path.size()) {
        cost = cost_t{0, 0} - geometry.dist(solution.path[insertion.indexes.first - 1],
                                                   solution.path[insertion.indexes.first]);
        cost += geometry.dist(solution.path[insertion.indexes.first - 1], insertion.segments.first)
                + geometry.cost(insertion.segments.first)
                + geometry.dist(insertion.segments.first, insertion.segments.second)
                + geometry.cost(insertion.segments.second)
                + geometry.dist(insertion.segments.second, solution.path[insertion.indexes.first]);
    } else {
        cost = geometry.dist(solution.path[insertion.indexes.first - 1], insertion.segments.first)
                + geometry.cost(insertion.segments.first)
                + geometry.dist(insertion.segments.first, insertion.segments.second)
                + geometry.cost(insertion.segments.second);
    }
    return cost;
}

void MinimumSegmentPairInsertion::correctIncumbentSegmentsOrder() {
    if (path_engine::is_node(incumbent.segments.second.source) && incumbent.indexes.first > incumbent.indexes.second){
        std::swap(incumbent.segments.first, incumbent.segments.second);
        std::swap(incumbent.indexes.first, incumbent.indexes.second);
    }
}


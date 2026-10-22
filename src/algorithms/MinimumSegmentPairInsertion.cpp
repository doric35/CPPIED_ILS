#include "../../include/algorithms/MinimumSegmentPairInsertion.hpp"

Insertion MinimumSegmentPairInsertion::solve() {
    const SegmentPair key{firstSegment, secondSegment};
    if (memoizedMinimalInsertions.contains(key))
        updateIncumbent(memoizedMinimalInsertions.at(key));
    else {
        if (solution.path.size() > 1) {
            solveDisjointInsertion();
            solvePairedInsertion();
        } else
            solveUnitPathEdgeCase();
    }
    correctIncumbentSegmentsOrder();
    return incumbent;
}

void MinimumSegmentPairInsertion::solveUnitPathEdgeCase() {
    for (auto &pair: pairedKernels) {
        cost_t cost = geometry.insert_tail_gain(solution.path.front(), pair.first);
        cost += geometry.insert_tail_gain(pair.first, pair.second);
        Insertion insertion{{1, 1}, cost, pair};
        updateIncumbent(insertion);
    }
}

void MinimumSegmentPairInsertion::solveDisjointInsertion() {
    MinimumTopTwoInsertion insertion_algorithm(instance, coverage, geometry, solution, firstSegment);
    TopTwoInsertion top2Insertions{};
    auto non_oriented = geometry.getNonOriented(firstSegment);
    auto key = encodeSegment(non_oriented.source, non_oriented.target);
    if (!memoizedDisjointTopInsertions.contains(key)) {
        top2Insertions = insertion_algorithm.solve();
        memoizedDisjointTopInsertions.insert({key, top2Insertions});
    }
    else {
        top2Insertions = memoizedDisjointTopInsertions.at(key);
    }
    TopTwoInsertion top2InsertionsOther{};
    non_oriented = geometry.getNonOriented(secondSegment);
    key = encodeSegment(non_oriented.source, non_oriented.target);
    if (!memoizedDisjointTopInsertions.contains(key)) {
        insertion_algorithm.resetSegment(secondSegment);
        top2InsertionsOther = insertion_algorithm.solve();
        memoizedDisjointTopInsertions.insert({key, top2InsertionsOther});
    } else {
        top2InsertionsOther = memoizedDisjointTopInsertions.at(key);
    }
    if (picksAreDominated(top2Insertions, top2InsertionsOther))
        throw DominatedInsertion("Segments can't be inserted at cost lower than the bound.");
    setIncumbentFromDisjointInsertions(top2Insertions, top2InsertionsOther);
}

bool MinimumSegmentPairInsertion::picksAreDominated(TopTwoInsertion& firstPick,
                                                    TopTwoInsertion& secondPick) {
    return std::max(firstPick.bestInsertion.insertion_cost, secondPick.bestInsertion.insertion_cost) >= bound;
}

void MinimumSegmentPairInsertion::setIncumbentFromDisjointInsertions(TopTwoInsertion& first_pick,
                                                                TopTwoInsertion& second_pick) {
    if (first_pick.bestInsertion.index == second_pick.bestInsertion.index &&
        first_pick.bestInsertion.insertion_cost + second_pick.backupInsertion.insertion_cost >
        first_pick.backupInsertion.insertion_cost + second_pick.bestInsertion.insertion_cost){
        std::swap(first_pick.bestInsertion, first_pick.backupInsertion);
    } else if (first_pick.bestInsertion.index == second_pick.bestInsertion.index)
        std::swap(second_pick.bestInsertion, second_pick.backupInsertion);
    Insertion trial{{first_pick.bestInsertion.index, second_pick.bestInsertion.index},
                    first_pick.bestInsertion.insertion_cost + second_pick.bestInsertion.insertion_cost,
                    {first_pick.bestInsertion.s, second_pick.bestInsertion.s}};
    updateIncumbent(trial);
}


void MinimumSegmentPairInsertion::solvePairedInsertion() {
    for (int i = 1; i <= solution.path.size(); i++)
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
    if (memoizedMinimalInsertions.contains(key) && memoizedMinimalInsertions.at(key).cost > trial.cost)
        memoizedMinimalInsertions[key] = trial;
    else
        memoizedMinimalInsertions.insert({key,trial});
}

cost_t MinimumSegmentPairInsertion::getInsertionCost(Insertion &insertion) {
    cost_t cost{};
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
    if (incumbent.indexes.first > incumbent.indexes.second){
        std::swap(incumbent.segments.first, incumbent.segments.second);
        std::swap(incumbent.indexes.first, incumbent.indexes.second);
    }
}


#include "../../include/heuristics/IntervalScheduler.hpp"

void IntervalScheduler::solve() {
    if (nIntervals == 0) return;
    else if (!solution.empty()) resetData();
    forwardPass();
    backwardPass();
}

void IntervalScheduler::resetData() {
    solution.clear();
    stateCosts = std::vector<cost_t>(nIntervals, cost_t{0,0});
    statePredecessors = std::vector<int>(nIntervals, -1);
}

void IntervalScheduler::forwardPass() {
    stateCosts[0] = intervals[0].gain;
    for (int i=1; i< nIntervals; i++)
        setStateCostAndPredecessor(i);
}

void IntervalScheduler::setStateCostAndPredecessor(int i) {
    auto itInterval = std::next(intervals.begin(), i);
    auto itPredecessor = getValidPredecessor(itInterval);
    if (itPredecessor != intervals.end())
        setStateWithPredecessor(itInterval, itPredecessor);
    else
        setStateWithoutPredecessor(itInterval);
}

IntervalSet::const_iterator IntervalScheduler::getValidPredecessor(IntervalSet::const_iterator itInterval) {
    int bound = itInterval->x1;
    auto onePastPredecessor = std::lower_bound(
            intervals.begin(), itInterval, bound, cmpEndPoint
            );
    if (onePastPredecessor != intervals.begin())
        return std::prev(onePastPredecessor);
    return intervals.end();

}

void IntervalScheduler::setStateWithoutPredecessor(IntervalSet::const_iterator itInterval) {
    std::size_t i = std::distance(intervals.begin(), itInterval);
    stateCosts[i] = std::max(stateCosts[i-1], itInterval->gain);
}

void IntervalScheduler::setStateWithPredecessor(IntervalSet::const_iterator itInterval, IntervalSet::const_iterator itPredecessor) {
    std::size_t i = std::distance(intervals.begin(), itInterval);
    std::size_t j = std::distance(intervals.begin(), itPredecessor);
    stateCosts[i] = std::max(stateCosts[i-1], itInterval->gain + stateCosts[j]);
    statePredecessors[i] = static_cast<int>(j);
}

void IntervalScheduler::backwardPass() {
    for (int i = nIntervals -1; i>=0;)
        i = setBackwardControl(i);
}

bool IntervalScheduler::isStateInSolution(IntervalSet::const_iterator itInterval) {
    std::size_t i = std::distance(intervals.begin(), itInterval);
    if (statePredecessors[i] >= 0)
        return itInterval->gain + stateCosts[statePredecessors[i]] == stateCosts[i];
    return itInterval->gain >= stateCosts[i];
}

bool IntervalScheduler::isPredecessorEquivalent(IntervalSet::const_iterator itInterval) {
    if (itInterval == intervals.begin()) return false;
    std::size_t i = std::distance(intervals.begin(), itInterval);
    if (isStateInSolution(std::prev(itInterval)))
        return stateCosts[i] == stateCosts[i-1];
    return false;
}

int IntervalScheduler::randomBackwardControl(IntervalSet::const_iterator itInterval) {
    std::size_t i = std::distance(intervals.begin(), itInterval);
    if (d(rng) < 0.5){
        solution.push(*itInterval);
        return statePredecessors[i];
    }
    return i-1;
}

int IntervalScheduler::setBackwardControl(int i) {
    const auto& state = std::next(intervals.begin(), i);
    bool state_is_candidate = isStateInSolution(state);
    if (state_is_candidate && isPredecessorEquivalent(state))
        i = randomBackwardControl(state);
    else if (state_is_candidate){
        solution.push(*state);
        i = statePredecessors[i];
    } else i--;
    return i;
}
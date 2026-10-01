#include "../../include/heuristics/IntervalScheduler.hpp"

void IntervalScheduler::solve() {
    if (nIntervals == 0) return;
    forwardPass();
    backwardPass();
}

void IntervalScheduler::forwardPass() {
    stateCosts[0] = begin->gain;
    for (int i=1; i< nIntervals; i++)
        setStateCostAndPredecessor(i);
}

void IntervalScheduler::setStateCostAndPredecessor(int i) {
    auto itInterval = std::next(begin, i);
    auto itPredecessor = getValidPredecessor(itInterval);
    if (isValidPredecessor(itPredecessor))
        setStateWithPredecessor(itInterval, itPredecessor);
    else
        setStateWithoutPredecessor(itInterval);
}

iVecIt IntervalScheduler::getValidPredecessor(iVecIt itInterval) {
    int bound = itInterval->x1;
    auto onePastPredecessor = std::lower_bound(
            begin, itInterval, bound, cmpEndPoint
            );
    if (onePastPredecessor != begin)
        return std::prev(onePastPredecessor);
    return end;

}

void IntervalScheduler::setStateWithoutPredecessor(iVecIt itInterval) {
    std::size_t i = std::distance(begin, itInterval);
    stateCosts[i] = std::max(stateCosts[i-1], itInterval->gain);
}

void IntervalScheduler::setStateWithPredecessor(iVecIt itInterval, iVecIt itPredecessor) {
    std::size_t i = std::distance(begin, itInterval);
    std::size_t j = std::distance(begin, itPredecessor);
    stateCosts[i] = std::max(stateCosts[i-1], itInterval->gain + stateCosts[j]);
    statePredecessors[i] = static_cast<int>(j);
}

void IntervalScheduler::backwardPass() {
    for (int i = nIntervals -1; i>=0;)
        i = setBackwardControl(i);
}

bool IntervalScheduler::isStateInSolution(iVecIt itInterval) {
    std::size_t i = std::distance(begin, itInterval);
    if (statePredecessors[i] >= 0)
        return itInterval->gain + stateCosts[statePredecessors[i]] == stateCosts[i];
    return itInterval->gain >= stateCosts[i];
}

bool IntervalScheduler::isPredecessorEquivalent(iVecIt itInterval) {
    std::size_t i = std::distance(begin, itInterval);
    if (isStateInSolution(std::prev(itInterval)))
        return stateCosts[i] == stateCosts[i-1];
    return false;
}

int IntervalScheduler::randomBackwardControl(iVecIt itInterval) {
    std::size_t i = std::distance(begin, itInterval);
    if (d(rng) < 0.5){
        solution.push_back(*itInterval);
        return statePredecessors[i];
    }
    return i-1;
}

int IntervalScheduler::setBackwardControl(int i) {
    const auto& state = std::next(begin, i);
    bool state_is_candidate = isStateInSolution(state);
    if (state_is_candidate && isPredecessorEquivalent(state))
        i = randomBackwardControl(state);
    else if (state_is_candidate){
        solution.push_back(*state);
        i = statePredecessors[i];
    } else i--;
    return i;
}
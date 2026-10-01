#include "../../include/heuristics/MaximumCover.hpp"

void MaximumCover::solve() {
    forwardPass();
    backwardPass();
}

void MaximumCover::forwardPass() {
    for (int i=0; i<minSpacing; i++)
        stateValues[i] = costs[i];

    for (int i = minSpacing; i< costs.size(); i++)
        stateValues[i] = std::max(stateValues[i-1], stateValues[i - minSpacing] + costs[i]);
}

void MaximumCover::backwardPass() {
    int i = backwardRecursion();
    auto max_item = std::max_element(costs.begin(), std::next(costs.begin(), i));
    i = std::distance(costs.begin(), max_item);
    solution.push_back(i);
}

int MaximumCover::backwardRecursion() {
    int i=0;
    for (i=costs.size() - 1; i>=minSpacing;)
        i = processCandidate(i);
    return i;
}

void MaximumCover::selectRoot(int i) {
    std::vector<int> candidates = setRootCandidates(i);
    int j = uniformChoice(candidates);
    solution.push_back(j);
}

int MaximumCover::processCandidate(int i) {
    if (isOptimalCandidate(i)) {
        std::vector<int> candidates = setRangeCandidates(i - minSpacing + 1, i);
        i = uniformChoice(candidates);
        solution.push_back(i);
        i -= minSpacing;
    }
    else i -= 1;
    return i;
}

std::vector<int> MaximumCover::setRangeCandidates(int j, int i) {
    std::vector<int> candidates;
    for (int k = j; k<=i; k++){
        if (isOptimalCandidate(k) && stateValues[k] == stateValues[i])
            candidates.push_back(k);
    }
    return candidates;
}

std::vector<int> MaximumCover::setRootCandidates(int i) {
    auto max_item = std::max_element(costs.begin(), std::next(costs.begin(), i+1));
    cost_t value = *max_item;
    std::vector<int> candidates(i+1, 0);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::erase_if(candidates, [&](int j){ return costs[j] != value; });
    return candidates;
}

int MaximumCover::uniformChoice(const std::vector<int> &range) {
    int choice;
    std::ranges::sample(range, &choice, 1, rng);
    return choice;
}

bool MaximumCover::isOptimalCandidate(int i) {
    return stateValues[i] == stateValues[i-minSpacing] + costs[i];
}
#include "../../include/heuristics/MaximumCover.hpp"

void MaximumCover::solve() {
    if (!solution.empty()) resetData();
    if (!costs.empty()) {
        forwardPass();
        backwardPass();
    }
}

void MaximumCover::resetData() {
    solution.clear();
    stateValues = std::vector<cost_t>(costs.size(), cost_t{0,0});
}

void MaximumCover::forwardPass() {
    stateValues[0] = costs[0];
    for (int i=1; i<minSpacing; i++)
        stateValues[i] = std::max(stateValues[i-1],costs[i]);

    for (int i = minSpacing; i< costs.size(); i++)
        stateValues[i] = std::max(stateValues[i-1], stateValues[i - minSpacing] + costs[i]);
}

void MaximumCover::backwardPass() {
    int i = backwardRecursion();
    if (i >= 0) solution.push_back(selectRoot(i));
}

int MaximumCover::backwardRecursion() {
    int i=0;
    for (i=costs.size() - 1; i>=minSpacing;)
        i = processCandidate(i);
    return i;
}

int MaximumCover::selectRoot(int i) {
    std::vector<int> candidates = setRootCandidates(i);
    return uniformChoice(candidates);
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
    std::vector<int> candidates(i+1, 0);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::erase_if(candidates, [&](int j){ return costs[j] != stateValues[i]; });
    return candidates;
}

int MaximumCover::uniformChoice(const std::vector<int> &range) {
    int choice;
    std::ranges::sample(range, &choice, 1, rng);
    return choice;
}

bool MaximumCover::isOptimalCandidate(int i) {
    if (i - minSpacing < 0) return stateValues[i] == costs[i];
    return stateValues[i] == stateValues[i-minSpacing] + costs[i];
}
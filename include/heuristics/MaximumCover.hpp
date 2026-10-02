#pragma once

#include "../structures.hpp"


class MaximumCover {
private:
    int minSpacing;

    std::uniform_real_distribution<> d{0.0, 1.0};
    std::mt19937 rng;

    const std::vector<cost_t>& costs;
    std::vector<cost_t> stateValues;
    std::vector<int> solution;

    void resetData();

    void forwardPass();
    void backwardPass();

    bool isOptimalCandidate(int i);
    std::vector<int> setRangeCandidates(int j, int i);
    std::vector<int> setRootCandidates(int i);
    int uniformChoice(const std::vector<int>& range);
    int processCandidate(int i);
    int backwardRecursion();
    int selectRoot(int i);
protected:

public:
    //----Constructors & Destructors----
    MaximumCover(const std::vector<cost_t>& costs, int spacing) :
        minSpacing(std::min(spacing, static_cast<int>(costs.size()))),
        rng(), costs(costs),
        stateValues(costs.size(), cost_t{0,0}),
        solution(){}

    ~MaximumCover() = default;

    //----Getters----
    std::vector<int> getSolution() { return solution; }
    cost_t getCost() {
        if (stateValues.empty()) return cost_t{0,0};
        return stateValues.back();
    }

    //----Setters----

    //----Member Functions----
    void solve();

    //----Static Functions----

    //----Friend Classes----
};

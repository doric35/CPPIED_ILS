#pragma once

#include <utility>
#include "../structures.hpp"

using iVecIt = std::vector<interval>::const_iterator;

class IntervalScheduler {
private:
    int nIntervals;
    std::uniform_real_distribution<> d{0.0, 1.0};
    std::mt19937 rng;

    std::vector<cost_t> stateCosts;
    std::vector<int> statePredecessors;
    std::vector<interval> solution;

    iVecIt begin, end;

    void forwardPass();
    void backwardPass();

    void setStateCostAndPredecessor(int i);
    iVecIt getValidPredecessor(iVecIt itInterval);
    bool isValidPredecessor(iVecIt itPredecessor){ return itPredecessor != end; }
    void setStateWithoutPredecessor(iVecIt itInterval);
    void setStateWithPredecessor(iVecIt itInterval, iVecIt itPredecessor);
    int randomBackwardControl(iVecIt itInterval);
    int setBackwardControl(int i);

    bool isStateInSolution(iVecIt itInterval);
    bool isPredecessorEquivalent(iVecIt itInterval);

protected:

public:
    //----Constructors & Destructors----
    IntervalScheduler(iVecIt begin, iVecIt end) :
        begin(begin), end(end),
        nIntervals(static_cast<int>(std::distance(begin, end))),
        stateCosts(nIntervals, cost_t{0,0}),
        statePredecessors(nIntervals, -1),
        solution(), rng(){}

    ~IntervalScheduler() = default;

    //----Getters----
    const std::vector<interval>& getSolution(){ return solution; }
    cost_t getCost(){ return stateCosts.back(); }

    //----Setters----

    //----Member Functions----
    void solve();

    //----Static Functions----
    static bool cmpEndPoint(const interval& interval, int bound){ return interval.x2 < bound; }

    //----Friend Classes----
};

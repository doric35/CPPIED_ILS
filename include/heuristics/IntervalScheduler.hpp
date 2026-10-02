#pragma once

#include "../data_structures/Interval.hpp"

using namespace cppied_intervals;

class IntervalScheduler {
private:
    int nIntervals;
    std::uniform_real_distribution<> d{0.0, 1.0};
    std::mt19937 rng;

    std::vector<cost_t> stateCosts;
    std::vector<int> statePredecessors;
    IntervalSet solution;

    const IntervalSet& intervals;

    void forwardPass();
    void backwardPass();

    void setStateCostAndPredecessor(int i);
    IntervalSet::const_iterator getValidPredecessor(IntervalSet::const_iterator itInterval);
    void setStateWithoutPredecessor(IntervalSet::const_iterator itInterval);
    void setStateWithPredecessor(IntervalSet::const_iterator itInterval, IntervalSet::const_iterator itPredecessor);
    int randomBackwardControl(IntervalSet::const_iterator itInterval);
    int setBackwardControl(int i);
    void resetData();

    bool isStateInSolution(IntervalSet::const_iterator itInterval);
    bool isPredecessorEquivalent(IntervalSet::const_iterator itInterval);

protected:

public:
    //----Constructors & Destructors----
    IntervalScheduler(const IntervalSet& intervals) :
        intervals(intervals),
        nIntervals(static_cast<int>(intervals.size())),
        stateCosts(nIntervals, cost_t{0,0}),
        statePredecessors(nIntervals, -1),
        solution(), rng(){}

    ~IntervalScheduler() = default;

    //----Getters----
    const IntervalSet& getSolution(){ return solution; }
    cost_t getCost(){
        cost_t cost{0,0};
        if (!stateCosts.empty()) cost = stateCosts.back();
        return cost; }

    //----Setters----

    //----Member Functions----
    void solve();

    //----Static Functions----
    static bool cmpEndPoint(const Interval& interval, int bound){ return interval.x2 < bound; }

    //----Friend Classes----
};

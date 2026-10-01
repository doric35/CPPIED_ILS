#pragma once

#include "IntervalScheduler.hpp"
#include "MaximumCover.hpp"

class ris_heuristic{
public:
    ris_heuristic(int pSpacing,
                  std::function<void(const std::vector<int>&,
                                     std::vector<std::vector<int>>&)> splitter,
                  std::function<void(std::vector<int>&)> filter,
                  std::function<void(const std::vector<int>&,
                                     std::vector<interval>&)> interval_transform) :
                  spacing(pSpacing),
                  schedules(),
                  rnd(),
                  rng(rnd()),
                  split_function(std::move(splitter)),
                  filter_function(std::move(filter)),
                  interval_function(std::move(interval_transform)){}
    void solve_selection(const std::vector<cost_t>& pGains,
                         std::vector<int>& pSelection);
    static int softmaxSample(std::vector<cost_t>& gains, std::mt19937& generator);
    static std::vector<double> setFlatScores(std::vector<cost_t> &gains);
    static bool cmpIntervalY (const interval& a, int y){ return a.y < y; };
    static bool cmpIntervalsPosition(const interval& a, const interval& b){
        if (a.y == b.y)
            return a.x2 < b.x2;
        return a.y < b.y;
    };
protected:

private:
    using iVecIt = std::vector<interval>::const_iterator;
    std::vector<std::vector<int>> setSegmentSets(const std::vector<cost_t> &pGains);
    std::vector<std::vector<interval>> setIntervalSets(std::vector<std::vector<int>>& candidateSets);
    void setNonOverlapIntervals(std::vector<std::vector<interval>>& intervalSets);
    std::vector<interval> setOptimalSchedule(std::vector<interval>& intervals);
    std::vector<std::pair<iVecIt, iVecIt>> groupIntervalsByRows(std::vector<interval>& intervals);
    std::vector<std::vector<interval>> setRowsOptimalSchedule(const std::vector<std::pair<iVecIt, iVecIt>>& intervalGroups);
    std::vector<interval> setRowOptimalSchedule(iVecIt begin, iVecIt end);
    std::vector<cost_t> setIntervalGains(std::vector<std::vector<interval>>& intervals);

    std::vector<int> setSegmentSelection(std::vector<cost_t>& gainSets);
    int softmaxSample(std::vector<cost_t>& gains);
    //others

    //Member variables
    int spacing;

    std::vector<std::vector<interval>> schedules;

    std::random_device rnd;
    std::mt19937 rng;

    std::function<void(const std::vector<int>&,
            std::vector<std::vector<int>>&)> split_function;
    std::function<void(std::vector<int>&)> filter_function;
    std::function<void(const std::vector<int>&,
            std::vector<interval>&)> interval_function;
};

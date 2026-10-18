#pragma once

#include "../algorithms/IntervalScheduler.hpp"
#include "../algorithms/MaximumCover.hpp"

class ris_heuristic{
public:
    ris_heuristic(int pSpacing, int size) : spacing(pSpacing), size(size), schedules(), rnd(), rng(rnd()){}
    std::vector<int> solve_selection(GroupedIntervalSets& buckets);
    static int softmaxSample(std::vector<cost_t>& gains, std::mt19937& generator);
    static std::vector<double> setFlatScores(std::vector<cost_t> &gains);

    void clear(){ schedules.clear(); }

    static IntervalSet getCoverIntervals(const std::vector<int>& cover, GroupedIntervalSets& intervals);
protected:

private:
    std::vector<int> getSelection(GroupedIntervalSets& intervalSets);

    [[nodiscard]] GroupedIntervalSets getRowsOptimalSchedule(const GroupedIntervalSets&) const;
    
    void setNonOverlappingGroups(GroupedIntervalSets& intervalSets);
    IntervalSet getOptimalSchedule(const GroupedIntervalSets& intervals);
    [[nodiscard]] IntervalSet getRowOptimalSchedule(const IntervalSet& intervals) const;

    std::vector<int> getSegmentSelection();
    std::vector<int> getCoverSolution(GroupedIntervalSets& intervals) const;
    int softmaxSample();
    //others

    //Member variables
    int spacing, size;

    GroupedIntervalSets schedules;

    std::random_device rnd;
    std::mt19937 rng;
};

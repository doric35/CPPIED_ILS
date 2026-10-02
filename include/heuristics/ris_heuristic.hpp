#pragma once

#include "IntervalScheduler.hpp"
#include "MaximumCover.hpp"

class ris_heuristic{
public:
    ris_heuristic(int pSpacing,
                  int size,
                  std::function<void(const std::vector<int>&, std::vector<std::vector<int>>&)> splitter,
                  std::function<void(std::vector<int>&)> filter,
                  std::function<void(const std::vector<int>&, IntervalSet&)> interval_transform) :
                  spacing(pSpacing),
                  size(size),
                  schedules(),
                  rnd(),
                  rng(rnd()),
                  split_function(std::move(splitter)),
                  filter_function(std::move(filter)),
                  interval_function(std::move(interval_transform)){}
    std::vector<int> solve_selection();
    static int softmaxSample(std::vector<cost_t>& gains, std::mt19937& generator);
    static std::vector<double> setFlatScores(std::vector<cost_t> &gains);

    void clear(){ schedules.clear(); }

    static IntervalSet getCoverIntervals(const std::vector<int>& cover, GroupedIntervalSets& intervals);
protected:

private:
    GroupedIntervalSets getBuckets();
    std::vector<int> getSelection(GroupedIntervalSets& intervalSets);

    std::vector<std::vector<int>> splitSegmentIds();
    GroupedIntervalSets getIntervalSets(std::vector<std::vector<int>>& candidateSets);

    [[nodiscard]] GroupedIntervalSets getRowsOptimalSchedule(const GroupedIntervalSets&) const;
    
    void setNonOverlappingGroups(GroupedIntervalSets& intervalSets);
    IntervalSet getOptimalSchedule(const GroupedIntervalSets& intervals);
    [[nodiscard]] IntervalSet getRowOptimalSchedule(const IntervalSet& intervals) const;

    std::vector<int> setSegmentSelection();
    std::vector<int> getCoverSolution(GroupedIntervalSets& intervals) const;
    int softmaxSample();
    //others

    //Member variables
    int spacing, size;

    GroupedIntervalSets schedules;

    std::random_device rnd;
    std::mt19937 rng;

    std::function<void(const std::vector<int>&, std::vector<std::vector<int>>&)> split_function;
    std::function<void(std::vector<int>&)> filter_function;
    std::function<void(const std::vector<int>&, IntervalSet&)> interval_function;
};

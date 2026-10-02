#include "../../include/heuristics/ris_heuristic.hpp"

std::vector<int> ris_heuristic::solve_selection() {
    if (!schedules.empty()) clear();
    GroupedIntervalSets buckets = getBuckets();
    return getSelection(buckets);
}

GroupedIntervalSets ris_heuristic::getBuckets() {
    std::vector<std::vector<int>> segment_ids_sets = splitSegmentIds();
    return getIntervalSets(segment_ids_sets);
}

std::vector<std::vector<int>> ris_heuristic::splitSegmentIds() {
    std::vector<int> candidates;
    candidates.resize(size);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::vector<std::vector<int>> candidate_sets;
    filter_function(candidates);
    split_function(candidates,candidate_sets);
    return candidate_sets;
}


std::vector<int> ris_heuristic::getSelection(GroupedIntervalSets& intervalSets){
    std::vector<int> selection;
    if (!intervalSets.empty()){
        setNonOverlappingGroups(intervalSets);
        if (!schedules.empty())
            selection = setSegmentSelection();
    }
    return selection;
}

GroupedIntervalSets ris_heuristic::getIntervalSets(std::vector<std::vector<int>>& candidateSets) {
    GroupedIntervalSets interval_sets;
    interval_sets.reserve(candidateSets.size());
    for (const auto & candidateSet : candidateSets) {
        IntervalSet intervals;
        interval_function(candidateSet, intervals);
        interval_sets.push(intervals);
    }
    return interval_sets;
}

void ris_heuristic::setNonOverlappingGroups(GroupedIntervalSets &intervalSets){
    schedules.reserve(intervalSets.size());
    std::vector<GroupedIntervalSets> collection = intervalSets.groupSetsByRows();
    for (auto & interval_set : collection) {
        schedules.push(getOptimalSchedule(interval_set));
        if (schedules.back().empty()) schedules.pop();
    }
}

IntervalSet ris_heuristic::getOptimalSchedule(const GroupedIntervalSets &intervals) {
    if (intervals.empty()) return {};
    auto solutions = getRowsOptimalSchedule(intervals);
    std::vector<int> cover_solution = getCoverSolution(solutions);
    return getCoverIntervals(cover_solution, solutions);
}

std::vector<int> ris_heuristic::getCoverSolution(GroupedIntervalSets &intervals) const {
    auto gains = intervals.getGroupWiseValues();
    MaximumCover solver(gains, static_cast<int>(spacing));
    solver.solve();
    return solver.getSolution();
}

IntervalSet ris_heuristic::getCoverIntervals(const std::vector<int> &cover, GroupedIntervalSets &intervals) {
    IntervalSet schedule;
    schedule.reserve(cover.size());
    for (int i: cover)
        schedule.push(intervals[i]);

    return schedule;
}

GroupedIntervalSets ris_heuristic::getRowsOptimalSchedule(const GroupedIntervalSets& collection) const {
    GroupedIntervalSets solutions;
    for (const auto& group : collection)
        solutions.push(getRowOptimalSchedule(group));
    return solutions;
}

IntervalSet ris_heuristic::getRowOptimalSchedule(const IntervalSet& intervals) const{
    IntervalScheduler solver(intervals);
    solver.solve();
    return solver.getSolution();
}

std::vector<int> ris_heuristic::setSegmentSelection() {
    std::vector<int> selection;
    int item = softmaxSample();
    selection.reserve(schedules[item].size());
    for (auto& i : schedules[item])
        selection.push_back(i.id);
    return selection;
}

int ris_heuristic::softmaxSample() {
    auto gains = schedules.getGroupWiseValues();
    return softmaxSample(gains, rng);
}

std::vector<double> ris_heuristic::setFlatScores(std::vector<cost_t> &gains) {
    auto cmpTurn = [](const cost_t& a, const cost_t& b){
        return a.turns < b.turns;
    };
    int max_turns = std::max_element(gains.begin(), gains.end(), cmpTurn)->turns;
    int M = max_turns + 1;
    std::vector<double> scores(gains.size());
    for (size_t i=0; i<gains.size(); i++)
        scores[i] = M * gains[i].length + gains[i].turns;
    return scores;
}

int ris_heuristic::softmaxSample(std::vector<cost_t> &gains, std::mt19937& generator) {
    std::vector<double> scores = setFlatScores(gains);
    double max_score = *std::max_element(scores.begin(), scores.end());
    auto softmax = [max_score](double score){
        return std::exp(score - max_score);
    };
    std::transform(scores.cbegin(), scores.cend(),scores.begin(), softmax);
    std::discrete_distribution<> d(scores.begin(),scores.end());
    return d(generator);
}
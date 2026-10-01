#include "../../include/heuristics/ris_heuristic.hpp"

void ris_heuristic::solve_selection(const std::vector<cost_t> &pGains,
                                    std::vector<int> &pSelection) {
    std::vector<std::vector<int>> candidate_sets = std::move(setSegmentSets(pGains));
    std::vector<std::vector<interval>> interval_sets = std::move(setIntervalSets(candidate_sets));

    if (interval_sets.empty()) return;

    setNonOverlapIntervals(interval_sets);
    std::vector<cost_t> gain_sets = setIntervalGains(schedules);

    if (!schedules.empty())
        pSelection = setSegmentSelection(gain_sets);
}

std::vector<std::vector<int>> ris_heuristic::setSegmentSets(const std::vector<cost_t> &pGains) {
    std::vector<int> candidates;
    candidates.resize(pGains.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    std::vector<std::vector<int>> candidate_sets;
    filter_function(candidates);
    split_function(candidates,candidate_sets);
    return candidate_sets;
}

std::vector<std::vector<interval>> ris_heuristic::setIntervalSets(std::vector<std::vector<int>>& candidateSets) {
    std::vector<std::vector<interval>> interval_sets(candidateSets.size());
    for (int i=0; i< candidateSets.size(); i++)
        interval_function(candidateSets[i], interval_sets[i]);
    return interval_sets;
}

void ris_heuristic::setNonOverlapIntervals(
        std::vector<std::vector<interval>> &intervalSets) {
    schedules.reserve(intervalSets.size());

    for (auto & intervalSet : intervalSets) {
        schedules.push_back(setOptimalSchedule(intervalSet));
        if (schedules.back().empty()) schedules.pop_back();
    }
}

std::vector<cost_t> ris_heuristic::setIntervalGains(std::vector<std::vector<interval>>& intervals) {
    std::vector<cost_t> gains(intervals.size(), cost_t{0,0});
    auto gainSum = [](cost_t acc, const interval& I){
        return acc + I.gain;
    };
    for (std::size_t i=0; i<intervals.size(); ++i)
        gains[i] = std::accumulate(intervals[i].begin(), intervals[i].end(),
                                   cost_t{0,0}, gainSum);
    return gains;
}

std::vector<interval> ris_heuristic::setOptimalSchedule(std::vector<interval> &intervals) {
    if (intervals.empty()) return {};

    std::sort(intervals.begin(), intervals.end(), cmpIntervalsPosition);
    auto rows = groupIntervalsByRows(intervals);
    auto solutions = setRowsOptimalSchedule(rows);
    auto gains = setIntervalGains(solutions);
    MaximumCover solver(gains, static_cast<int>(spacing));
    solver.solve();
    std::vector<int> cover_solution = solver.getSolution();

    std::vector<interval> schedule;
    schedule.reserve(cover_solution.size());
    for (int i: cover_solution)
        schedule.insert(schedule.end(), schedules[i].begin(), schedules[i].end());

    return schedule;
}

std::vector<std::pair<iVecIt, iVecIt>> ris_heuristic::groupIntervalsByRows(std::vector<interval> &intervals) {
    std::sort(intervals.begin(), intervals.end(), cmpIntervalsPosition);
    int min_y = intervals.front().y;
    int max_y = intervals.back().y;
    std::vector<std::pair<iVecIt, iVecIt>> rows(max_y - min_y + 1, {intervals.end(), intervals.end()});
    for (auto source = intervals.begin(); source != intervals.end();){
        auto end = std::lower_bound(
                source, intervals.end(),source->y + 1, cmpIntervalY);
        rows[source->y - min_y] = {source, end};
        source = end;
    }
    return rows;
}

std::vector<std::vector<interval>> ris_heuristic::setRowsOptimalSchedule(
        const std::vector<std::pair<iVecIt, iVecIt>>& intervalGroups) {
    std::vector<std::vector<interval>> solutions;
    for (auto& group : intervalGroups)
        solutions.push_back(setRowOptimalSchedule(group.first, group.second));
    return solutions;
}

std::vector<interval> ris_heuristic::setRowOptimalSchedule(ris_heuristic::iVecIt begin, ris_heuristic::iVecIt end) {
    std::vector<interval> solution;
    if (begin != end) {
        IntervalScheduler solver(begin, end);
        solver.solve();
        solution = solver.getSolution();
    }
    return solution;
}

std::vector<int> ris_heuristic::setSegmentSelection(std::vector<cost_t> &gainSets) {
    std::vector<int> selection;
    int item = softmaxSample(gainSets);
    selection.reserve(schedules[item].size());
    for (auto& i : schedules[item])
        selection.push_back(i.id);
    return selection;
}

int ris_heuristic::softmaxSample(std::vector<cost_t> &gains) {
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
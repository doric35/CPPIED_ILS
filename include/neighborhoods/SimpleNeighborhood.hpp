#pragma once

#include "neighborhood.hpp"

template<typename Move, typename MoveSearchAlgorithm>
class SimpleNeighborhood : public neighborhood {
private:

protected:
    int spacingConstraint = 2;
    std::vector<int> splitIndices{};
    std::vector<Move> candidatesMoves{};
    virtual void clear(){
        splitIndices.clear();
        candidatesMoves.clear();
    }
    virtual void applyMoves(std::vector<int>& selection) = 0;
    Move searchMove(int i){
        MoveSearchAlgorithm algorithm(problem, coverage, geometry, solution, i);
        return algorithm.solve();
    }

    void setMovesAndGains(){
        candidatesMoves.resize(solution.path.size());
        candidatesMoves[0].gain = cost_t{MIN_INT,MIN_INT};
        for(int i=1; i< solution.path.size(); ++i)
            candidatesMoves[i] = searchMove(i);
    }
    std::vector<int> selectMoves(){
        std::vector<int> candidates(candidatesMoves.size(), 0);
        std::iota(candidates.begin(), candidates.end(), 0);
        pruneNonImprovingMoves(candidates);
        splitMoves(candidates);
        GroupedIntervalSets collection = constructIntervalSets(candidates);
        ris_heuristic selection_heuristic(spacingConstraint*problem.max_range,static_cast<int>(candidatesMoves.size()));
        return selection_heuristic.solve_selection(collection);
    }
    void pruneNonImprovingMoves(std::vector<int>& candidates){
        cost_t bound = {0,0};
        std::erase_if(candidates, [&](int i){ return candidatesMoves[i].gain <= bound; });
    }
    GroupedIntervalSets constructIntervalSets(std::vector<int> &candidates) {
        GroupedIntervalSets grouped;
        int begin = 0;
        for (int split : splitIndices) {
            grouped.push(constructIntervalSet(
                    std::next(candidates.begin(), begin), std::next(candidates.begin(), split)));
            begin = split;
        }
        grouped.push(constructIntervalSet(
                std::next(candidates.begin(), begin), candidates.end()));
        return grouped;
    }
    IntervalSet constructIntervalSet(std::vector<int>::iterator begin, std::vector<int>::iterator end) {
        IntervalSet intervals;
        for (auto it = begin; it!= end; ++it)
            intervals.push(transformToInterval(*it));
        return intervals;
    }
    virtual Interval transformToInterval(int i) {
        segment non_oriented = geometry.getNonOriented(solution.path[i]);
        int x1 = non_oriented.source;
        int x2 = path_engine::is_node(non_oriented.target) ?
                 non_oriented.target : non_oriented.source;
        int y;
        bool horizontal = geometry.is_horizontal(non_oriented);
        if (horizontal) {
            y = geometry.get_row(non_oriented);
            x1 = x1 % geometry.n_cols;
            x2 = x2 % geometry.n_cols;
        }
        else {
            y = geometry.get_column(non_oriented);
            x1 = (x1 - geometry.horizontal_bound) % geometry.n_rows;
            x2 = (x2 - geometry.horizontal_bound) % geometry.n_rows;
        }
        return Interval{cost_t{0,0}, x1 , x2, y, !horizontal, i};
    }
    virtual void splitMoves(std::vector<int>& candidates) = 0;
    void splitOddEvenMoves(std::vector<int> &candidates) {
        auto odd_even_split = [](int i){
            return i % 2 == 0;
        };
        auto it = std::stable_partition(candidates.begin(), candidates.end(), odd_even_split);
        splitIndices = {static_cast<int>(std::distance(candidates.begin(), it))};
    }

    void splitHorizontalVerticalMoves(std::vector<int> &candidates) {
        if (splitIndices.empty())
            splitIndices.push_back(splitHorizontalVerticalSubset(candidates.begin(), candidates.end()));
        else
            splitHorizontalVerticalNonEmptySplit(candidates);
    }

    int splitHorizontalVerticalSubset(std::vector<int>::iterator begin, std::vector<int>::iterator end){
        auto horizontal_vertical_split = [&](int i){ return geometry.is_horizontal(solution.path[i]); };
        auto it = std::stable_partition(begin, end, horizontal_vertical_split);
        return std::distance(begin, it);
    }

    void splitHorizontalVerticalNonEmptySplit(std::vector<int>& candidates){
        std::vector<int> new_split; new_split.reserve(2*splitIndices.size());
        int begin = 0;
        for (int i : splitIndices){
            int relative_split = splitHorizontalVerticalSubset(candidates.begin() + begin, candidates.begin() + i);
            new_split.push_back(relative_split + begin);
            begin = i;
            new_split.push_back(i);
        }
        int relative_split = splitHorizontalVerticalSubset(candidates.begin() + begin, candidates.end());
        new_split.push_back(relative_split + begin);
        splitIndices.swap(new_split);
    }
public:
    //----Constructors & Destructors----
    using neighborhood::neighborhood;
    ~SimpleNeighborhood() override = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    bool local_search() override{
        if (path_engine::is_node(solution.path[0].target)){
            std::pair<segment, segment> split = geometry.splitSegment(solution.path[0], 0);
            solution.path[0] = split.first;
            solution.path.insert(solution.path.begin() + 1, split.second);
        }
        setMovesAndGains();
        std::vector<int> move_selection = selectMoves();
        applyMoves(move_selection);
        clear();
        return !move_selection.empty();
    }

    //----Static Functions----

    //----Friend Classes----
};

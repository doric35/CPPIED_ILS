#pragma once

#include "../structures.hpp"
#include "../cppied_method_base.hpp"

template<typename Move>
class MaximumMove {
private:

protected:
    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;
    int index;

    std::vector<Move> candidates;
    Move incumbent;

    void setPositiveCandidates();
    void setCandidatesGains();
    virtual void setIncumbentMove();

    segment trimSegmentOvercoveredHeadAndTail(const segment& s);
    segment trimSegmentHead(const segment& s);
    segment trimSegmentTail(const segment& s);

    virtual bool isValidMove(Move& move) = 0;
    virtual cost_t getGain(Move& move) = 0;

public:
    //----Constructors & Destructors----
    MaximumMove(cppied_instance& instance,
                coverage_engine& coverage,
                path_engine&     geometry,
                cppied_solution& solution,
                int index) : instance(instance),
                             coverage(coverage), geometry(geometry),
                             solution(solution), index(index),
                             incumbent(){
        incumbent.gain = {MIN_INT, MIN_INT};
    }

    ~MaximumMove() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    virtual Move solve() = 0;

    //----Static Functions----

    //----Friend Classes----
};

template<typename Move>
void MaximumMove<Move>::setCandidatesGains() {
    for (auto& move : candidates)
        move.gain = getGain(move);
}

template<typename Move>
void MaximumMove<Move>::setPositiveCandidates(){
    std::erase_if(candidates, [](const Move& move){ return move.gain <= cost_t{0,0};});
}

template<typename Move>
void MaximumMove<Move>::setIncumbentMove() {
    auto cmpGains = [](
            const Move& a, const Move& b){
        return a.gain > b.gain;
    };
    std::sort(candidates.begin(), candidates.end(), cmpGains);
    for (auto& move : candidates){
        if (isValidMove(move)){
            incumbent = move;
            return;
        }
    }
}

template<typename Move>
segment MaximumMove<Move>::trimSegmentOvercoveredHeadAndTail(const segment& s){
    segment trimmed = geometry.getNonOriented(s);
    if (!path_engine::is_node(trimmed.target) && coverage.can_remove_vertex(solution, trimmed.source))
        trimmed = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    else if (path_engine::is_node(trimmed.target)) {
        trimmed = trimSegmentHead(trimmed);
        if (trimmed.source != trimmed.target)
            trimmed = trimSegmentTail(trimmed);
        if (trimmed.source == trimmed.target)
            trimmed = {trimmed.source, path_engine::NULL_NODE};
    }
    if (path_engine::is_node(trimmed.source))
        geometry.correct_segment_direction(trimmed, geometry.get_direction(s));
    return trimmed;
}

template<typename Move>
segment MaximumMove<Move>::trimSegmentHead(const segment& s){
    segment trimmed{path_engine::NULL_NODE, path_engine::NULL_NODE};

    auto nodes = std::views::iota(s.source, s.target + 1);

    auto isRemovable = [&](int node){
        return coverage.can_remove_vertex(solution, node);
    };

    auto first_non_removable = std::ranges::find_if_not(nodes, isRemovable);

    if (first_non_removable != nodes.end())
        trimmed = {*first_non_removable, s.target};

    return trimmed;
}

template<typename Move>
segment MaximumMove<Move>::trimSegmentTail(const segment& s){
    segment trimmed{path_engine::NULL_NODE, path_engine::NULL_NODE};

    auto nodes = std::views::iota(s.source, s.target + 1);
    auto reversed = nodes | std::views::reverse;

    auto isRemovable = [&](int node){
        return coverage.can_remove_vertex(solution, node);
    };

    auto last_non_removable = std::ranges::find_if_not(reversed, isRemovable);

    if (last_non_removable != reversed.end())
        trimmed = {s.source, *last_non_removable};

    return trimmed;
}

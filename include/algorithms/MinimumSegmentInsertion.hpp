#pragma once

#include "MaximumMove.hpp"

struct SegmentInsertion{
    segment s;
    int index;
    cost_t insertion_cost;
};

class MinimumSegmentInsertion {
private:
    SegmentInsertion incumbent;
protected:
    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;
    segment nonOrientedSegment;

    void solveNonTerminalInsertions();
    void solveTerminalInsertions();

public:
    //----Constructors & Destructors----
    MinimumSegmentInsertion(cppied_instance& instance,
                            coverage_engine& coverage,
                            path_engine&     geometry,
                            cppied_solution& solution,
                            segment s) :
                            incumbent(), instance(instance),
                            coverage(coverage), geometry(geometry),
                            solution(solution), nonOrientedSegment(geometry.getNonOriented(s)){
        incumbent.insertion_cost = {MAX_INT, MAX_INT};
    }

    ~MinimumSegmentInsertion() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    SegmentInsertion solve();

    //----Static Functions----

    //----Friend Classes----
};

#pragma once

#include "MinimumSegmentInsertion.hpp"

struct TopTwoInsertion{
    SegmentInsertion bestInsertion;
    SegmentInsertion backupInsertion;
};

class MinimumTopTwoInsertion {
private:

protected:
    TopTwoInsertion incumbent;

    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;
    segment nonOrientedSegment;

    void solveNonTerminalInsertions();
    void solveTerminalInsertions();
    void updateIncumbent(SegmentInsertion& move);
public:
    //----Constructors & Destructors----
    MinimumTopTwoInsertion(cppied_instance& instance,
                           coverage_engine& coverage,
                           path_engine&     geometry,
                           cppied_solution& solution,
                           segment s) : incumbent(), instance(instance),
                           coverage(coverage), geometry(geometry),
                           solution(solution), nonOrientedSegment(geometry.getNonOriented(s)){
        incumbent.bestInsertion.insertion_cost = {MAX_INT, MAX_INT};
        incumbent.backupInsertion.insertion_cost = {MAX_INT, MAX_INT};
    }

    ~MinimumTopTwoInsertion() = default;

    //----Getters----

    //----Setters----
    void resetSegment(segment& s){
        nonOrientedSegment = geometry.getNonOriented(s);
        incumbent.bestInsertion.insertion_cost = {MAX_INT,MAX_INT};
        incumbent.backupInsertion.insertion_cost = {MAX_INT, MAX_INT};
    }

    //----Member Functions----
    TopTwoInsertion solve();

    //----Static Functions----

    //----Friend Classes----
};

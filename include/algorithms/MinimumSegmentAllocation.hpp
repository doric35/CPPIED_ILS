#pragma once

#include "../structures.hpp"
#include "../utils.hpp"
#include "../cppied_method_base.hpp"

struct Allocation{
    segment s;
    int index;
    cost_t cost;
};

class MinimumSegmentAllocation {
private:
    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;

    Allocation incumbent;
    std::vector<segment> candidates;
    std::vector<int> unsatisfiedCells;
    iRectangle unsatisfiedBox;
protected:
    void setUnsatisfiedCells();
    void setCompleteCoverageSegments();
    void setCompleteCoverageSegmentsHorizontal();
    void setCompleteCoverageSegmentsVertical();
    void setUnsatisfiedRectangle();
    void filterCandidates();
    bool isValidAllocation(const segment& s);
    void updateIncumbent(segment& s);
    cost_t getGain(const segment& s);
public:
    //----Constructors & Destructors----
    MinimumSegmentAllocation(cppied_instance& instance,
                             coverage_engine& coverage,
                             path_engine&     geometry,
                             cppied_solution& solution) : 
                             instance(instance), coverage(coverage), geometry(geometry), solution(solution),
                             incumbent(), unsatisfiedCells(), unsatisfiedBox(){
        incumbent.s = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        incumbent.cost = {MAX_INT, MAX_INT};
        incumbent.index = -1;
    }

    ~MinimumSegmentAllocation() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    Allocation solve();

    //----Static Functions----

    //----Friend Classes----
};

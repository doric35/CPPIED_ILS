#pragma once

#include "../structures.hpp"
#include "../cppied_method_base.hpp"

struct SegmentPacking{
    Eigen::VectorXi mask;
    std::vector<int> items;
};

class RandomSegmentPacking{
private:
    SegmentPacking packing;
protected:
    cppied_instance& instance;
    coverage_engine& coverage;
    path_engine&     geometry;
    cppied_solution& solution;
    std::vector<int> candidates;

    std::random_device rnd;
    std::mt19937 rng;

    void packShuffledCandidates();

public:
    RandomSegmentPacking(cppied_instance& instance, coverage_engine& coverage, path_engine& geometry,
                         cppied_solution& solution, std::vector<int>& candidates) : packing(), instance(instance),
                         coverage(coverage), geometry(geometry), solution(solution), candidates(candidates),
                         rnd(), rng(rnd()){}

    std::vector<int> solve();
};
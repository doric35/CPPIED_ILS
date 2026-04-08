#pragma once

#include <utility>

#include "../structures.hpp"

class ris_heuristic{
public:
    ris_heuristic(int pSpacing,
                  std::function<void(const std::vector<int>&,
                                     std::vector<std::vector<int>>&)> splitter,
                  std::function<void(std::vector<int>&)> filter,
                  std::function<void(const std::vector<int>&,
                                     std::vector<interval>&)> interval_transform) :
                  spacing(pSpacing),
                  split_function(std::move(splitter)),
                  filter_function(std::move(filter)),
                  interval_function(std::move(interval_transform)),
                  rnd(),
                  rng(rnd()){}
    void solve_selection(const std::vector<cost_t>& pGains,
                         std::vector<int>& pSelection);
    static int softmax_sample(std::vector<cost_t>& gains,
                              std::mt19937& generator);
private:
    //others
    using iVecIt = std::vector<interval>::const_iterator;

    //Member functions
    void set_selection(
            std::vector<interval>&,
            std::vector<interval>&,
            cost_t&);
    void spacing_dp(
            const std::vector<cost_t>&,
            std::vector<int>&,
            cost_t&);
    void interval_dp(
            iVecIt begin, iVecIt end,
            std::vector<interval>&,
            cost_t&);
    int softmax_sample(
            std::vector<cost_t>& gains,
            std::vector<bool>& no_goods
            );

    //Member variables
    int spacing;

    std::random_device rnd;
    std::mt19937 rng;

    std::function<void(const std::vector<int>&,
            std::vector<std::vector<int>>&)> split_function;
    std::function<void(std::vector<int>&)> filter_function;
    std::function<void(const std::vector<int>&,
            std::vector<interval>&)> interval_function;
};

#pragma once

#include "neighborhood.hpp"
#include "neighborhood_n12.hpp"
#include "neighborhood_n21.hpp"
#include "../algorithms/RandomOneTwoReallocation.hpp"

class neighborhood_nested : public neighborhood {
private:
    std::random_device rd;
    std::mt19937 rng;

    std::vector<cost_t> removalGains;
    int randomMoveIndex;

    cppied_solution trial;

    void setRemovalGains();
    OneTwoReallocationMove getRandomMove();
    void applyRandomMove(OneTwoReallocationMove& move);
    void n21_search();
protected:
public:
    template<typename... Args>
    neighborhood_nested(Args&&... args) : neighborhood(std::forward<Args>(args)...), rd(),
        rng(rd()), removalGains(), trial(solution), randomMoveIndex(-1){}

    bool local_search() override;
};
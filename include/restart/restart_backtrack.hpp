#pragma once

#include "Restart.hpp"
#include "../construction/dp_sweeper.hpp"

class restart_backtrack : public  Restart{
public:
    using  Restart::Restart;
    void restart(cppied_solution&) override;
    void sample(cppied_solution&);
protected:
    std::queue<std::vector<bool>> history{};
    std::vector<segment> warm_start{};
};